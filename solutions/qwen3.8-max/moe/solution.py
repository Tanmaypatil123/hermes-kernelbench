"""Sonic-MoE up-projection: variable-length grouped GEMM + fused SwiGLU.

Custom Triton implementation (no vendor GEMM dispatch). Structure:

1. `_tile_map_kernel` — tiny single-program kernel turning int32
   `expert_offsets` prefix sums into a flat tile table
   (tile_expert, tile_row0, tile_end) for every (expert, BM-row-block) pair,
   padded with -1 sentinels up to MAX_TILES = cdiv(T_perm, BM) + E.
   Everything stays on-device (no host sync, CUDA-graph friendly).

2. `_moe_up_kernel` — the grouped GEMM. Each program computes BOTH fp32
   accumulators (gate and up) over the full K loop with a single shared A
   tile, then fuses h = silu(acc_gate) * acc_u in registers and stores bf16
   directly. Grid order makes the N-tile the fast-varying dimension so a
   launch wave covers all N columns of the same few experts: weight slices
   are reused across an expert's M-tiles from L2, A-tiles across N columns.
   Sentinel programs exit immediately, so ragged / empty expert slices cost
   nothing.

The best (BM, BN, BK, warps, stages) is picked once per process per (H, I)
by an on-device timing sweep; every candidate is verified against a proven
baseline config before it may win, so an exotic config can never trade
correctness for speed.

Note: RTX PRO 6000 (SM120) runs bf16 mma with fp32 accumulate at full rate
(503.8 TFLOPS dense), so fp16 accumulation offers no speed advantage.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import triton
import triton.language as tl

OP_TYPE = "grouped_gemm_swiglu"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]

# Sweep candidates. Dual fp32 accumulators (gate+up) of BMxBN need
# BM*BN*8 bytes of registers, e.g. 128x128 -> 128 regs/thread at 8 warps.
_CONFIGS = [
    {"BM": 128, "BN": 128, "BK": 64, "warps": 8, "stages": 2},
    {"BM": 128, "BN": 64, "BK": 64, "warps": 4, "stages": 3},
    {"BM": 128, "BN": 64, "BK": 32, "warps": 4, "stages": 4},
    {"BM": 128, "BN": 128, "BK": 32, "warps": 8, "stages": 4},
    {"BM": 256, "BN": 64, "BK": 32, "warps": 8, "stages": 4},
    {"BM": 64, "BN": 64, "BK": 64, "warps": 4, "stages": 4},
]

_CFG_CACHE: dict = {}
_VER_PRINTED = [False]


@triton.jit
def _tile_map_kernel(
    offsets_ptr,        # (E+1,) int32 prefix sums
    te_ptr,             # (MAX_TILES,) int32 out: expert id, -1 = sentinel
    tr_ptr,             # (MAX_TILES,) int32 out: first row of this tile
    tend_ptr,           # (MAX_TILES,) int32 out: end row of this expert slice
    E,
    MAX_TILES,
    BM: tl.constexpr,
    NT_BLOCK: tl.constexpr,   # >= cdiv(T_perm, BM), power of two
    E_BLOCK: tl.constexpr,    # >= E, power of two
):
    idx = tl.arange(0, NT_BLOCK)
    tile = 0
    for e in range(E):
        start = tl.load(offsets_ptr + e)
        end = tl.load(offsets_ptr + e + 1)
        nt = tl.cdiv(end - start, BM)
        m = idx < nt
        tl.store(te_ptr + tile + idx, e, mask=m)
        tl.store(tr_ptr + tile + idx, start + idx * BM, mask=m)
        tl.store(tend_ptr + tile + idx, end, mask=m)
        tile += nt
    # Pad the tail with sentinels so the GEMM grid can be a static upper bound.
    eidx = tl.arange(0, E_BLOCK)
    tl.store(te_ptr + tile + eidx, -1, mask=eidx < MAX_TILES - tile)


@triton.jit
def _moe_up_kernel(
    x_ptr,              # (T_perm, H) bf16, permuted hidden states
    wg_ptr,             # (E, H, I) bf16 gate weights
    wu_ptr,             # (E, H, I) bf16 up weights
    out_ptr,            # (T_perm, I) bf16
    te_ptr, tr_ptr, tend_ptr,
    H, I,  # noqa: E741
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    e = tl.load(te_ptr + pid_m)
    if e < 0:
        return
    row0 = tl.load(tr_ptr + pid_m)
    rend = tl.load(tend_ptr + pid_m)

    rm = row0 + tl.arange(0, BM)
    rmask = rm < rend
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    nmask = rn < I

    a_ptrs = x_ptr + rm[:, None] * H + rk[None, :]
    w_off = e * (H * I) + rk[:, None] * I + rn[None, :]
    g_ptrs = wg_ptr + w_off
    u_ptrs = wu_ptr + w_off

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(H, BK)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=rmask[:, None], other=0.0)
        else:
            kmask = (k * BK + rk) < H
            a = tl.load(a_ptrs, mask=rmask[:, None] & kmask[None, :], other=0.0)
        if EVEN_N:
            g = tl.load(g_ptrs)
            u = tl.load(u_ptrs)
        else:
            g = tl.load(g_ptrs, mask=nmask[None, :], other=0.0)
            u = tl.load(u_ptrs, mask=nmask[None, :], other=0.0)
        acc_g = tl.dot(a, g, acc_g)
        acc_u = tl.dot(a, u, acc_u)
        a_ptrs += BK
        g_ptrs += BK * I
        u_ptrs += BK * I

    # Fused SwiGLU epilogue: silu(gate) * up, in fp32, rounded once.
    res = acc_g * tl.sigmoid(acc_g) * acc_u
    o_ptrs = out_ptr + rm[:, None] * I + rn[None, :]
    if EVEN_N:
        tl.store(o_ptrs, res.to(out_ptr.dtype.element_ty), mask=rmask[:, None])
    else:
        tl.store(o_ptrs, res.to(out_ptr.dtype.element_ty),
                 mask=rmask[:, None] & nmask[None, :])


def _next_pow2(n: int) -> int:
    return max(1, 1 << (max(int(n), 1) - 1).bit_length())


class Model(nn.Module):
    """Up-projection of a top-K MoE FFN with fused SwiGLU.

    Inputs at call time:
      hidden_states:    (T_perm, H)  bf16, already permuted to expert order
      expert_offsets:   (E+1,)       int32 prefix sums; expert e owns rows
                        [offsets[e]:offsets[e+1]]
    Output:
      gated_up:         (T_perm, I)  bf16
    """

    def __init__(self, T_total: int, H: int, I: int, E: int, K: int):  # noqa: E741
        super().__init__()
        self.T_total = T_total
        self.H = H
        self.I = I
        self.E = E
        self.K = K
        self.W_gate = nn.Parameter(torch.empty(E, H, I, dtype=torch.bfloat16))
        self.W_up = nn.Parameter(torch.empty(E, H, I, dtype=torch.bfloat16))
        nn.init.normal_(self.W_gate, std=0.02)
        nn.init.normal_(self.W_up, std=0.02)
        # Plain python attrs -> invisible to state_dict (strict load stays OK).
        self._bufs: dict = {}

    def _get_bufs(self, device, BM: int, max_tiles: int):
        key = (str(device), BM, max_tiles)
        bufs = self._bufs.get(key)
        if bufs is None:
            bufs = (
                torch.empty(max_tiles, dtype=torch.int32, device=device),
                torch.empty(max_tiles, dtype=torch.int32, device=device),
                torch.empty(max_tiles, dtype=torch.int32, device=device),
            )
            self._bufs[key] = bufs
        return bufs

    def _build_map(self, offsets, device, T_perm: int, BM: int):
        """(Re)build the tile table for row-tile size BM. Returns
        (te, tr, tend, max_tiles). Buffers cached; kernel runs every call."""
        max_tiles = (T_perm + BM - 1) // BM + self.E
        te, tr, tend = self._get_bufs(device, BM, max_tiles)
        _tile_map_kernel[(1,)](
            offsets, te, tr, tend, self.E, max_tiles,
            BM=BM,
            NT_BLOCK=_next_pow2((T_perm + BM - 1) // BM),
            E_BLOCK=_next_pow2(self.E),
            num_warps=4,
        )
        return te, tr, tend, max_tiles

    def _launch_gemm(self, cfg, x, out, te, tr, tend, max_tiles, H, I):
        BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
        grid = (triton.cdiv(I, BN), max_tiles)
        _moe_up_kernel[grid](
            x, self.W_gate, self.W_up, out, te, tr, tend, H, I,
            BM=BM, BN=BN, BK=BK,
            EVEN_K=(H % BK == 0), EVEN_N=(I % BN == 0),
            num_warps=cfg["warps"], num_stages=cfg["stages"],
        )

    def _pick_cfg(self, x, out, offsets, H, I, T_perm):
        key = (H, I)
        cfg = _CFG_CACHE.get(key)
        if cfg is not None:
            return cfg
        if not _VER_PRINTED[0]:
            _VER_PRINTED[0] = True
            print(f"SOLCFG triton={triton.__version__} torch={torch.__version__}",
                  flush=True)
        base = _CONFIGS[0]  # proven-correct fallback (validated on GPU twice)
        best = base
        use_cuda = x.is_cuda and not os.environ.get("TRITON_INTERPRET")
        if use_cuda:
            from triton.testing import do_bench
            # Tile maps for every candidate BM, built once (buffers cached,
            # contents valid for this call's offsets).
            maps = {}
            for c in _CONFIGS:
                if c["BM"] not in maps:
                    maps[c["BM"]] = self._build_map(offsets, x.device,
                                                    T_perm, c["BM"])

            def _bench(c):
                te, tr, tend, mt = maps[c["BM"]]
                return do_bench(
                    lambda: self._launch_gemm(c, x, out, te, tr, tend, mt,
                                              H, I),
                    warmup=25, rep=75)

            te, tr, tend, mt = maps[base["BM"]]
            self._launch_gemm(base, x, out, te, tr, tend, mt, H, I)
            torch.cuda.synchronize()
            ref = out.clone()
            best_t = _bench(base)
            print(f"SOLCFG cand0 {base} ms={best_t:.4f}", flush=True)
            for cand in _CONFIGS[1:]:
                try:
                    te, tr, tend, mt = maps[cand["BM"]]
                    self._launch_gemm(cand, x, out, te, tr, tend, mt, H, I)
                    torch.cuda.synchronize()
                    o = out.float()
                    d = (o - ref.float()).abs().max().item()
                    if torch.isnan(o).any().item() or not (d < 0.01):
                        print(f"SOLCFG skip(mismatch {d:.3e}) {cand}", flush=True)
                        continue
                    t = _bench(cand)
                except Exception as ex:
                    print(f"SOLCFG skip({type(ex).__name__}: {str(ex)[:160]}) "
                          f"{cand}", flush=True)
                    continue
                print(f"SOLCFG cand {cand} ms={t:.4f}", flush=True)
                if t < best_t * 0.995:  # require a real margin to switch
                    best, best_t = cand, t
            print(f"SOLCFG H={H} I={I} chose {best} ms={best_t:.4f}", flush=True)
        _CFG_CACHE[key] = best
        return best

    def forward(
        self,
        hidden_states: torch.Tensor,   # (T_perm, H) bf16
        expert_offsets: torch.Tensor,  # (E+1,) int32
    ) -> torch.Tensor:
        T_perm, H = hidden_states.shape
        I = self.I  # noqa: E741
        hidden_states = hidden_states.contiguous()
        expert_offsets = expert_offsets.contiguous()
        out = torch.empty((T_perm, I), dtype=torch.bfloat16,
                          device=hidden_states.device)
        if T_perm == 0:
            return out
        cfg = _CFG_CACHE.get((H, I))
        if cfg is None:
            # First call for this (H, I): build a default-BM map and sweep.
            te, tr, tend, mt = self._build_map(expert_offsets, out.device,
                                               T_perm, 128)
            cfg = self._pick_cfg(hidden_states, out, expert_offsets, H, I,
                                 T_perm)
        te, tr, tend, mt = self._build_map(expert_offsets, out.device,
                                           T_perm, cfg["BM"])
        self._launch_gemm(cfg, hidden_states, out, te, tr, tend, mt, H, I)
        return out


# ---------------------------------------------------------------------------
# Module-level shape shims (mirror reference.py; check.py/benchmark.py may
# rewrite these per shape).
# ---------------------------------------------------------------------------
T_total = 32768
H = 4096
I = 1536  # noqa: E741
E = 128
K = 8


def _build_routing(T_total: int, E: int, K: int, device: str = "cpu") -> torch.Tensor:
    T_perm = T_total * K
    base = T_perm // E
    rem = T_perm - base * E
    counts = torch.full((E,), base, dtype=torch.int32, device=device)
    counts[:rem] += 1
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def get_inputs():
    T_perm = T_total * K
    hidden_states = torch.randn(T_perm, H, dtype=torch.bfloat16) * 0.1
    expert_offsets = _build_routing(T_total, E, K)
    return [hidden_states, expert_offsets]


def get_init_inputs():
    return [T_total, H, I, E, K]
