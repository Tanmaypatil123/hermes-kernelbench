"""Fused W4A16 weight-only quantized GEMM (AWQ/GPTQ asymmetric int4) for SM120.

Design:
- Packed uint8 weights (K/2, N) are viewed as int32 (K/2, N/4): one word holds
  2 k-values x 4 n-values. Loads are 16B-vectorized; nibbles are unpacked in
  registers (GPTQ/AWQ-style), dequantized per group (128) with bf16 scales and
  integer zero-points, and accumulated on tensor cores (fp32).
- SPLIT_K with an fp32 partials buffer + tiny reduce kernel keeps all SMs busy
  for decode shapes (M=1, small N).
- Autotune on first call per shape, then frozen direct launches (low CPU overhead).
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


def _pack_int4(w_q: torch.Tensor) -> torch.Tensor:
    K, N = w_q.shape
    assert K % 2 == 0
    lo = w_q[0::2].to(torch.uint8) & 0xF
    hi = w_q[1::2].to(torch.uint8) & 0xF
    return (lo | (hi << 4)).contiguous()


def _unpack_int4(w_packed: torch.Tensor, K: int) -> torch.Tensor:
    Kh, N = w_packed.shape
    assert Kh * 2 == K
    out = torch.empty((K, N), dtype=torch.uint8, device=w_packed.device)
    out[0::2] = w_packed & 0xF
    out[1::2] = (w_packed >> 4) & 0xF
    return out


def _prune(configs, named_args, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    bm_max = max(16, triton.next_power_of_2(M))
    out = []
    for c in configs:
        BM = c.kwargs["BLOCK_M"]
        BN = c.kwargs["BLOCK_N"]
        stages = c.num_stages
        if BM > bm_max or BN > N or BN < 64:
            continue
        # smem per stage: x 2*(BM*64*2)B ; v 64*BN B ; s/z 4*BN B
        smem = stages * (BM * 256 + 68 * BN)
        if smem > 96 * 1024:
            continue
        out.append(c)
    if not out:
        out = [configs[0]]
    return out


def _cfg(BM, BN, warps, stages):
    return triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_warps=warps, num_stages=stages)


_CONFIGS = [
    _cfg(16, 64, 4, 3), _cfg(16, 64, 4, 4), _cfg(16, 64, 4, 6),
    _cfg(16, 64, 8, 4), _cfg(16, 128, 4, 4), _cfg(16, 128, 8, 3),
    _cfg(16, 128, 8, 4), _cfg(16, 256, 8, 3),
    _cfg(32, 64, 4, 4), _cfg(32, 128, 4, 4), _cfg(32, 128, 8, 4),
    _cfg(32, 256, 8, 3),
    _cfg(64, 64, 4, 4), _cfg(64, 128, 4, 3), _cfg(64, 128, 4, 4),
    _cfg(64, 128, 8, 3), _cfg(64, 256, 8, 2), _cfg(64, 256, 8, 3),
    _cfg(128, 64, 4, 3), _cfg(128, 128, 8, 2), _cfg(128, 128, 8, 3),
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "SPLIT_K"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _w4a16_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_wk, stride_sg,
    stride_ys, stride_ym,
    GROUPS_PER_SPLIT: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    FP32_OUT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    BLOCK_J: tl.constexpr = BLOCK_N // 4  # int32 columns per block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = pid_n * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_kh = tl.arange(0, GROUP_SIZE_K // 2)
    m_mask = offs_m[:, None] < M
    n0 = offs_j * 4

    acc0 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)

    g0 = pid_k * GROUPS_PER_SPLIT
    for gi in range(GROUPS_PER_SPLIT):
        g = g0 + gi
        k0 = g * GROUP_SIZE_K
        kh0 = k0 // 2

        w_ptrs = w_ptr + (kh0 + offs_kh)[:, None] * stride_wk + offs_j[None, :]
        if EVEN_N:
            v = tl.load(w_ptrs, eviction_policy="evict_first")
        else:
            v = tl.load(w_ptrs, mask=offs_j[None, :] * 4 < N, other=0,
                        eviction_policy="evict_first")

        x_base = x_ptr + offs_m[:, None] * stride_xm
        if EVEN_M:
            x_e = tl.load(x_base + (k0 + 2 * offs_kh)[None, :])
            x_o = tl.load(x_base + (k0 + 2 * offs_kh + 1)[None, :])
        else:
            x_e = tl.load(x_base + (k0 + 2 * offs_kh)[None, :], mask=m_mask, other=0.0)
            x_o = tl.load(x_base + (k0 + 2 * offs_kh + 1)[None, :], mask=m_mask, other=0.0)

        n0 = offs_j * 4
        if EVEN_N:
            s0 = tl.load(s_ptr + g * stride_sg + n0).to(tl.float32)
            z0 = tl.load(z_ptr + g * stride_sg + n0).to(tl.float32)
            s1 = tl.load(s_ptr + g * stride_sg + n0 + 1).to(tl.float32)
            z1 = tl.load(z_ptr + g * stride_sg + n0 + 1).to(tl.float32)
            s2 = tl.load(s_ptr + g * stride_sg + n0 + 2).to(tl.float32)
            z2 = tl.load(z_ptr + g * stride_sg + n0 + 2).to(tl.float32)
            s3 = tl.load(s_ptr + g * stride_sg + n0 + 3).to(tl.float32)
            z3 = tl.load(z_ptr + g * stride_sg + n0 + 3).to(tl.float32)
        else:
            nmask0 = n0 < N
            nmask1 = n0 + 1 < N
            nmask2 = n0 + 2 < N
            nmask3 = n0 + 3 < N
            s0 = tl.load(s_ptr + g * stride_sg + n0, mask=nmask0, other=0.0).to(tl.float32)
            z0 = tl.load(z_ptr + g * stride_sg + n0, mask=nmask0, other=0.0).to(tl.float32)
            s1 = tl.load(s_ptr + g * stride_sg + n0 + 1, mask=nmask1, other=0.0).to(tl.float32)
            z1 = tl.load(z_ptr + g * stride_sg + n0 + 1, mask=nmask1, other=0.0).to(tl.float32)
            s2 = tl.load(s_ptr + g * stride_sg + n0 + 2, mask=nmask2, other=0.0).to(tl.float32)
            z2 = tl.load(z_ptr + g * stride_sg + n0 + 2, mask=nmask2, other=0.0).to(tl.float32)
            s3 = tl.load(s_ptr + g * stride_sg + n0 + 3, mask=nmask3, other=0.0).to(tl.float32)
            z3 = tl.load(z_ptr + g * stride_sg + n0 + 3, mask=nmask3, other=0.0).to(tl.float32)

        # byte i of the int32 word holds columns n = 4j + i (low nibble k=2r, high k=2r+1)
        lo0 = ((v & 0xF).to(tl.float32) - z0[None, :]) * s0[None, :]
        hi0 = (((v >> 4) & 0xF).to(tl.float32) - z0[None, :]) * s0[None, :]
        v1 = v >> 8
        lo1 = ((v1 & 0xF).to(tl.float32) - z1[None, :]) * s1[None, :]
        hi1 = (((v1 >> 4) & 0xF).to(tl.float32) - z1[None, :]) * s1[None, :]
        v2 = v >> 16
        lo2 = ((v2 & 0xF).to(tl.float32) - z2[None, :]) * s2[None, :]
        hi2 = (((v2 >> 4) & 0xF).to(tl.float32) - z2[None, :]) * s2[None, :]
        v3 = v >> 24
        lo3 = ((v3 & 0xF).to(tl.float32) - z3[None, :]) * s3[None, :]
        hi3 = (((v3 >> 4) & 0xF).to(tl.float32) - z3[None, :]) * s3[None, :]

        acc0 = tl.dot(x_e, lo0.to(tl.bfloat16), acc0)
        acc0 = tl.dot(x_o, hi0.to(tl.bfloat16), acc0)
        acc1 = tl.dot(x_e, lo1.to(tl.bfloat16), acc1)
        acc1 = tl.dot(x_o, hi1.to(tl.bfloat16), acc1)
        acc2 = tl.dot(x_e, lo2.to(tl.bfloat16), acc2)
        acc2 = tl.dot(x_o, hi2.to(tl.bfloat16), acc2)
        acc3 = tl.dot(x_e, lo3.to(tl.bfloat16), acc3)
        acc3 = tl.dot(x_o, hi3.to(tl.bfloat16), acc3)

    y_base = y_ptr + pid_k * stride_ys + offs_m[:, None] * stride_ym
    c0 = n0[None, :]
    if EVEN_M and EVEN_N:
        if FP32_OUT:
            tl.store(y_base + c0, acc0)
            tl.store(y_base + c0 + 1, acc1)
            tl.store(y_base + c0 + 2, acc2)
            tl.store(y_base + c0 + 3, acc3)
        else:
            tl.store(y_base + c0, acc0.to(y_ptr.dtype.element_ty))
            tl.store(y_base + c0 + 1, acc1.to(y_ptr.dtype.element_ty))
            tl.store(y_base + c0 + 2, acc2.to(y_ptr.dtype.element_ty))
            tl.store(y_base + c0 + 3, acc3.to(y_ptr.dtype.element_ty))
    else:
        st = m_mask & (c0 < N)
        if FP32_OUT:
            tl.store(y_base + c0, acc0, mask=st)
            tl.store(y_base + c0 + 1, acc1, mask=st & (c0 + 1 < N))
            tl.store(y_base + c0 + 2, acc2, mask=st & (c0 + 2 < N))
            tl.store(y_base + c0 + 3, acc3, mask=st & (c0 + 3 < N))
        else:
            tl.store(y_base + c0, acc0.to(y_ptr.dtype.element_ty), mask=st)
            tl.store(y_base + c0 + 1, acc1.to(y_ptr.dtype.element_ty), mask=st & (c0 + 1 < N))
            tl.store(y_base + c0 + 2, acc2.to(y_ptr.dtype.element_ty), mask=st & (c0 + 2 < N))
            tl.store(y_base + c0 + 3, acc3.to(y_ptr.dtype.element_ty), mask=st & (c0 + 3 < N))


@triton.jit
def _reduce_kernel(
    yp_ptr, y_ptr, MN, stride_ys,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(yp_ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(yp_ptr + s * stride_ys + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, acc.to(tl.bfloat16), mask=mask)


class Model(nn.Module):
    """W4A16 GEMM: y = x @ dequant(w_q, scales, zeros)."""

    def __init__(self, M: int, N: int, K: int, group_size: int = GROUP_SIZE):
        super().__init__()
        assert K % group_size == 0, "K must be divisible by group_size"
        assert K % 2 == 0, "K must be even (int4 packing)"
        self.M, self.N, self.K = M, N, K
        self.group_size = group_size
        n_groups = K // group_size

        # Same synthetic quant as the reference; harness may overwrite via state_dict.
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
        self.register_buffer("scales", scales.squeeze(1).to(torch.bfloat16).contiguous())
        self.register_buffer("zeros", zeros.squeeze(1).to(torch.bfloat16).contiguous())

        self._plan_cache: dict = {}
        self._yp_cache: dict = {}
        self._nsm: int | None = None
        self._w32 = None

    def _w32_view(self):
        if self._w32 is None:
            self._w32 = self.w_q.view(torch.int32)
        return self._w32

    # ------------------------------------------------------------------
    def _pick_split_k(self, M: int, N: int, K: int) -> int:
        if self._nsm is None:
            self._nsm = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        bm = 16 if M <= 16 else (32 if M <= 64 else 64)
        nprog = triton.cdiv(N, 64) * triton.cdiv(M, bm)
        groups = K // self.group_size
        sk = 1
        while nprog * sk < 2 * self._nsm and sk < 8 and groups % (sk * 2) == 0:
            sk *= 2
        return sk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M, K = x.shape
        N = self.scales.shape[1]

        plan = self._plan_cache.get(M)
        if plan is None:
            plan = self._tune(x, M, N, K)
        grid, BM, BN, SK, even_m, even_n, warps, stages, gps = plan

        w32 = self._w32_view()
        sxm = x.stride(0)
        swk = w32.stride(0)
        ssg = self.scales.stride(0)
        y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        if SK == 1:
            _w4a16_kernel.fn[grid](
                x, w32, self.scales, self.zeros, y,
                M, N, K,
                sxm, swk, ssg, 0, N,
                GROUPS_PER_SPLIT=gps, GROUP_SIZE_K=self.group_size, SPLIT_K=SK,
                BLOCK_M=BM, BLOCK_N=BN,
                EVEN_M=even_m, EVEN_N=even_n, FP32_OUT=False,
                num_warps=warps, num_stages=stages,
            )
            return y

        key = (M, N, SK)
        yp = self._yp_cache.get(key)
        if yp is None:
            yp = torch.empty((SK, M * N), dtype=torch.float32, device=x.device)
            self._yp_cache[key] = yp
        _w4a16_kernel.fn[grid](
            x, w32, self.scales, self.zeros, yp,
            M, N, K,
            sxm, swk, ssg, M * N, N,
            GROUPS_PER_SPLIT=gps, GROUP_SIZE_K=self.group_size, SPLIT_K=SK,
            BLOCK_M=BM, BLOCK_N=BN,
            EVEN_M=even_m, EVEN_N=even_n, FP32_OUT=True,
            num_warps=warps, num_stages=stages,
        )
        MN = M * N
        RBLOCK = 1024
        _reduce_kernel[(triton.cdiv(MN, RBLOCK),)](
            yp, y, MN, MN, SPLIT_K=SK, BLOCK=RBLOCK, num_warps=4, num_stages=1,
        )
        return y

    def _tune(self, x, M, N, K):
        SK = self._pick_split_k(M, N, K)
        groups = K // self.group_size
        gps = groups // SK
        y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        if SK == 1:
            out, ys, ym, fp32 = y, 0, N, False
        else:
            yp = torch.empty((SK, M * N), dtype=torch.float32, device=x.device)
            self._yp_cache[(M, N, SK)] = yp
            out, ys, ym, fp32 = yp, M * N, N, True

        def _grid(META):
            return (triton.cdiv(N, META["BLOCK_N"]), triton.cdiv(M, META["BLOCK_M"]), SK)

        _w4a16_kernel[_grid](
            x, self._w32_view(), self.scales, self.zeros, out,
            M, N, K,
            x.stride(0), self._w32_view().stride(0), self.scales.stride(0),
            ys, ym,
            GROUPS_PER_SPLIT=gps, GROUP_SIZE_K=self.group_size, SPLIT_K=SK,
            EVEN_M=False, EVEN_N=False, FP32_OUT=fp32,
        )
        best = _w4a16_kernel.best_config
        if hasattr(best, "kwargs"):
            BM, BN = best.kwargs["BLOCK_M"], best.kwargs["BLOCK_N"]
            warps, stages = best.num_warps, best.num_stages
        else:
            BM, BN = best["BLOCK_M"], best["BLOCK_N"]
            warps, stages = best.get("num_warps", 4), best.get("num_stages", 3)
        plan = (
            (triton.cdiv(N, BN), triton.cdiv(M, BM), SK),
            BM, BN, SK, (M % BM) == 0, (N % BN) == 0, warps, stages, gps,
        )
        self._plan_cache[M] = plan
        print(f"[w4a16] tuned M={M} N={N} K={K} -> BM={BM} BN={BN} SK={SK} "
              f"warps={warps} stages={stages}", flush=True)
        if SK == 1:
            return plan
        # produce a correct y for this (tuning) call
        MN = M * N
        _reduce_kernel[(triton.cdiv(MN, 1024),)](
            self._yp_cache[(M, N, SK)], y, MN, MN, SPLIT_K=SK, BLOCK=1024,
            num_warps=4, num_stages=1,
        )
        return plan


M = 1
N = 12288
K = 4096


def get_inputs():
    x = torch.randn(M, K, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [M, N, K]
