"""Fused W4A16 (int4 weight, bf16 activation) GEMM in Triton.

Weight layout: w_q is (K//2, N) uint8, low nibble = even-K row, high = odd.
A K-block of 128 rows (one quant group) is 64 packed bytes, so the K-reduction
splits into an even-half and an odd-half tl.dot against strided views of x --
unpack, dequant and GEMM are fused in one pass, no (K, N) bf16 weight is ever
materialized:

  out[m,n] = sum_k x[m,k] * ((w4[k,n] - z[k//128,n]) * s[k//128,n])

On first forward the packed weights are repacked once into a tile-contiguous
layout ((k_block, n_block, HK, BN)) so every weight-tile load is one
contiguous, fully-coalesced 4 KB (or 8 KB) DRAM burst. Split-K partial sums
are reduced with a preallocated fp32 buffer to keep decode launch overhead
minimal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

OP_TYPE = "gemm_w4a16"
SUPPORTED_PRECISIONS = ["int4_bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]

GROUP_SIZE = 128




@triton.jit
def _w4a16_gemm(
    X, WQ, S, Z, OUT, PART,
    M, N, K, N_NB,
    sxm, sxk,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GS: tl.constexpr,
    SPLIT: tl.constexpr, K_PER: tl.constexpr, DIRECT: tl.constexpr,
    TILED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)      # global N index
    rnl = tl.arange(0, BN)                  # index within the tile
    m_ok = rm < M
    n_ok = rn < N

    HK: tl.constexpr = BK // 2  # packed rows per K block

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k0 = pid_k * K_PER

    for it in range(0, K_PER, BK):
        kb = k0 + it
        g = kb // GS

        # activations: even and odd K lanes
        ke = kb + 2 * tl.arange(0, HK)
        xe = tl.load(X + rm[:, None] * sxm + ke[None, :] * sxk,
                     mask=m_ok[:, None], other=0.0)
        xo = tl.load(X + rm[:, None] * sxm + (ke + 1)[None, :] * sxk,
                     mask=m_ok[:, None], other=0.0)

        # packed weights (HK, BN) uint8 -> int4 nibbles
        if TILED:
            rb = tl.arange(0, HK)
            wbase = ((kb // BK) * N_NB + pid_n) * (HK * BN)
            q = tl.load(WQ + wbase + rb[:, None] * BN + rnl[None, :],
                        mask=n_ok[None, :], other=0).to(tl.int32)
            sbase = g * (N_NB * BN) + pid_n * BN
            s = tl.load(S + sbase + rnl, mask=n_ok, other=0.0).to(tl.float32)
            z = tl.load(Z + sbase + rnl, mask=n_ok, other=0.0).to(tl.float32)
        else:
            rb = kb // 2 + tl.arange(0, HK)
            q = tl.load(WQ + rb[:, None] * N + rn[None, :] * 1,
                        mask=n_ok[None, :], other=0).to(tl.int32)
            s = tl.load(S + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            z = tl.load(Z + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)

        lo = (q & 0xF)
        hi = (q >> 4) & 0xF

        wl = ((lo - z[None, :]) * s[None, :]).to(tl.bfloat16)
        wh = ((hi - z[None, :]) * s[None, :]).to(tl.bfloat16)

        acc += tl.dot(xe, wl)
        acc += tl.dot(xo, wh)

    if DIRECT:
        tl.store(OUT + rm[:, None] * N + rn[None, :],
                 acc.to(tl.bfloat16), mask=m_ok[:, None] & n_ok[None, :])
    else:
        tl.store(PART + pid_k.to(tl.int64) * M * N + rm[:, None] * N + rn[None, :],
                 acc, mask=m_ok[:, None] & n_ok[None, :])


@triton.jit
def _reduce_split(P, O, MN, N, SPLIT: tl.constexpr, BLK: tl.constexpr):
    """Sum fp32 split-K partials over dim 0 and cast to bf16, one launch."""
    pid = tl.program_id(0)
    idx = pid * BLK + tl.arange(0, BLK)
    ok = idx < MN
    acc = tl.zeros((BLK,), dtype=tl.float32)
    for i in tl.static_range(SPLIT):
        acc += tl.load(P + i * MN + idx, mask=ok, other=0.0)
    tl.store(O + idx, acc.to(tl.bfloat16), mask=ok)


def _pick_cfg(M: int, N: int, K: int, n_sm: int):
    BM = 16 if M <= 16 else (32 if M <= 32 else 64)
    BN = 128 if M >= 32 else 64
    BK = GROUP_SIZE
    n_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    split = 1
    if n_tiles < 2 * n_sm:
        want = (2 * n_sm + n_tiles - 1) // n_tiles
        while split < want and split < K // BK and split * 2 <= 32:
            split *= 2
    return BM, BN, BK, split


def _repack(w_q, scales, zeros, K, N, BN, BK):
    """Tile-contiguous layout: wq (KB, NB, HK, BN), s/z (G, NB, BN)."""
    HK = BK // 2
    n_kb = K // BK
    n_nb = N // BN
    wq_t = (w_q.view(n_kb, HK, n_nb, BN)
              .permute(0, 2, 1, 3).contiguous())
    s_t = scales.view(K // BK, n_nb, BN).contiguous()
    z_t = zeros.view(K // BK, n_nb, BN).contiguous()
    return (wq_t, s_t, z_t)


_BUF_CACHE: dict = {}


def _buf(key, shape, dtype, device):
    b = _BUF_CACHE.get(key)
    if b is None or b.shape != shape or b.device != device:
        b = torch.empty(shape, dtype=dtype, device=device)
        _BUF_CACHE[key] = b
    return b


def _w4a16_forward(x, w_q, scales, zeros, M, N, K, group_size, model=None):
    x = x.to(torch.bfloat16)
    if not x.is_contiguous():
        x = x.contiguous()
    n_sm = torch.cuda.get_device_properties(x.device).multi_processor_count
    BM, BN, BK, split = _pick_cfg(M, N, K, n_sm)
    K_PER = K // split
    assert K_PER % BK == 0 and BK == group_size

    tiled = (N % BN == 0) and (K // BK) * (N // BN) > 0 and model is not None
    if tiled:
        if getattr(model, "_tile_cache", None) is None:
            model._tile_cache = {}
        ckey = ("t", N, BN, BK)
        packed = model._tile_cache.get(ckey)
        if packed is None:
            packed = _repack(w_q, scales, zeros, K, N, BN, BK)
            model._tile_cache[ckey] = packed
        wq_k, s_k, z_k = packed
        n_nb = N // BN
    else:
        wq_k, s_k, z_k, n_nb = w_q, scales, zeros, 0

    out = _buf(("out", x.device, M, N), (M, N), torch.bfloat16, x.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), split)
    if split == 1:
        _w4a16_gemm[grid](
            x, wq_k, s_k, z_k, out, out,
            M, N, K, n_nb,
            x.stride(0), x.stride(1),
            BM=BM, BN=BN, BK=BK, GS=group_size,
            SPLIT=split, K_PER=K_PER, DIRECT=True,
            TILED=tiled,
            num_warps=4 if BM * BN <= 2048 else 8,
            num_stages=4 if BM > 16 else 6,
        )
        return out
    part = _buf(("part", x.device, M, N, split), (split, M, N),
                torch.float32, x.device)
    _w4a16_gemm[grid](
        x, wq_k, s_k, z_k, out, part,
        M, N, K, n_nb,
        x.stride(0), x.stride(1),
        BM=BM, BN=BN, BK=BK, GS=group_size,
        SPLIT=split, K_PER=K_PER, DIRECT=False,
        TILED=tiled,
        num_warps=4 if BM * BN <= 2048 else 8,
        num_stages=4 if BM > 16 else 6,
    )
    MN = M * N
    _reduce_split[(triton.cdiv(MN, 1024),)](
        part, out, MN, N, SPLIT=split, BLK=1024, num_warps=4)
    return out


def _pack_int4(w_q: torch.Tensor) -> torch.Tensor:
    K, N = w_q.shape
    lo = w_q[0::2].to(torch.uint8) & 0xF
    hi = w_q[1::2].to(torch.uint8) & 0xF
    return (lo | (hi << 4)).contiguous()


class Model(nn.Module):
    """W4A16 GEMM: y = x @ dequant(w_q, scales, zeros)."""

    def __init__(self, M: int, N: int, K: int, group_size: int = GROUP_SIZE):
        super().__init__()
        assert K % group_size == 0
        assert K % 2 == 0
        self.M, self.N, self.K = M, N, K
        self.group_size = group_size
        n_groups = K // group_size

        torch.manual_seed(0xC0DE ^ (M * 1315423911 + N * 2654435761 + K))
        w_full = torch.randn(K, N, dtype=torch.float32) * 0.02
        w_g = w_full.view(n_groups, group_size, N)
        w_min = w_g.min(dim=1, keepdim=True).values
        w_max = w_g.max(dim=1, keepdim=True).values
        scales = (w_max - w_min).clamp_min(1e-8) / 15.0
        zeros = (-w_min / scales).round().clamp(0, 15)
        w_q = ((w_g / scales) + zeros).round().clamp(0, 15).to(torch.uint8)
        w_q = w_q.view(K, N)

        self.register_buffer("w_q", _pack_int4(w_q))
        self.register_buffer("scales", scales.squeeze(1).to(torch.bfloat16))
        self.register_buffer("zeros", zeros.squeeze(1).to(torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _w4a16_forward(
            x, self.w_q, self.scales, self.zeros,
            x.shape[0], self.N, self.K, self.group_size, self)


M = 1
N = 12288
K = 4096


def get_inputs():
    x = torch.randn(M, K, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [M, N, K]
