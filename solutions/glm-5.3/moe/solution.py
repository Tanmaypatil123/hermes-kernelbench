"""Grouped GEMM + fused SwiGLU up-projection for top-K MoE (Triton).

Per expert e: h_e = silu(x_e @ W_gate[e]) * (x_e @ W_up[e]).

Design:
 * W_gate and W_up are interleaved into one (E, H, 2I) tensor (cached on
   (data_ptr, version)), so gate and up are computed by a SINGLE tl.dot with a
   single (BLOCK_M, 2*BLOCK_N) accumulator living in Blackwell TMEM, and the
   SwiGLU epilogue splits it via tl.split -- fully fused, one pass.
 * Variable-length experts are handled inside the kernel: each program derives
   its expert and global row range from expert_offsets with a couple of vector
   loads + reductions, so no per-expert launches, no padding, no host sync,
   and BLOCK_M stays a free autotune parameter.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

OP_TYPE = "grouped_gemm_swiglu"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]


def _cfg(BM, BN, BK, stages, warps):
    return triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
        num_stages=stages, num_warps=warps)


# All configs fit the ~99KB shared-memory budget; BLOCK sizes are for the
# gate/up-interleaved single-dot kernel (B tile is BLOCK_K x 2*BLOCK_N).
_AUTOTUNE_CONFIGS = [
    _cfg(128, 128, 64, 2, 8),
    _cfg(128, 128, 32, 3, 8),
    _cfg(128, 256, 32, 2, 8),
    _cfg(64, 128, 64, 3, 8),
    _cfg(256, 128, 32, 2, 8),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["H", "I"],
                 warmup=3, rep=5, use_cuda_graph=False)
@triton.jit
def _grouped_swiglu_kernel(
    X, WINT, OUT,
    EO,                 # (E+1,) int32 row prefix sums
    E,                  # runtime expert count
    H, I,
    stride_xm,
    stride_w_e, stride_w_h,
    stride_om,
    EP2: tl.constexpr,     # pow2 >= E+1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    n_tiles_n = tl.cdiv(I, BLOCK_N)
    pid = tl.program_id(0)
    pid_m = pid // n_tiles_n
    pid_n = pid % n_tiles_n

    # ---- expert lookup (BM-agnostic) ----
    offs_e = tl.arange(0, EP2)
    eoA = tl.load(EO + offs_e, mask=offs_e < E, other=0)          # EO[0..E-1]
    eoB = tl.load(EO + offs_e + 1, mask=offs_e < E, other=0)      # EO[1..E]
    counts = tl.where(offs_e < E, eoB, 0) - tl.where(offs_e < E, eoA, 0)
    ntiles = (counts + (BLOCK_M - 1)) // BLOCK_M
    csum_incl = tl.cumsum(ntiles, 0)                # inclusive tile prefix sums
    e = tl.sum((csum_incl <= pid_m).to(tl.int32))   # expert of this tile
    tiles_before = csum_incl - ntiles               # exclusive prefix at each expert
    local = pid_m - tl.sum(tl.where(offs_e == e, tiles_before, 0))
    total_tiles = tl.sum(ntiles)
    if pid_m >= total_tiles:
        return
    m_start = tl.load(EO + e)
    m_end = tl.load(EO + e + 1)
    row_start = m_start + local * BLOCK_M

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < m_end
    mask_n = offs_n < I

    acc = tl.zeros((BLOCK_M, 2 * BLOCK_N), dtype=tl.float32)

    Xp = X + offs_m[:, None] * stride_xm
    Wp = WINT + e.to(tl.int64) * stride_w_e
    # interleaved B tile: columns 2*i (+1 for up) of expert e
    offs_c = 2 * pid_n * BLOCK_N + tl.arange(0, 2 * BLOCK_N)
    Bp = Wp + offs_c[None, :]

    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(Xp + offs_k[None, :],
                    mask=mask_m[:, None] & (offs_k[None, :] < H), other=0.0)
        b = tl.load(Bp + offs_k[:, None] * stride_w_h,
                    mask=(offs_c // 2) < I, other=0.0)
        acc += tl.dot(a, b)

    gu = tl.reshape(acc, (BLOCK_M, BLOCK_N, 2))
    gate, up = tl.split(gu)
    silu = gate * tl.sigmoid(gate)
    out = (silu * up).to(OUT.dtype.element_ty)
    Om = OUT + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(Om, out, mask=mask_m[:, None] & mask_n[None, :])


class Model(nn.Module):
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
        self._w_cache = None  # (key, interleaved tensor)

    def _interleaved(self):
        wg, wu = self.W_gate, self.W_up
        key = (wg.data_ptr(), wg._version, wu.data_ptr(), wu._version)
        if self._w_cache is None or self._w_cache[0] != key:
            wi = torch.stack((wg.detach(), wu.detach()), dim=-1)  # (E,H,I,2)
            self._w_cache = (key, wi.view(self.E, self.H, 2 * self.I))
        return self._w_cache[1]

    def forward(self, hidden_states, expert_offsets):
        out = torch.empty(hidden_states.shape[0], self.I,
                          dtype=torch.bfloat16, device=hidden_states.device)
        x = hidden_states.contiguous()
        eo = expert_offsets.contiguous()
        w = self._interleaved()

        grid = ((x.shape[0] // 64 + self.E + 1) * triton.cdiv(self.I, 128),)
        _grouped_swiglu_kernel[grid](
            x, w, out,
            eo, self.E, self.H, self.I,
            x.stride(0),
            w.stride(0), w.stride(1),
            out.stride(0),
            EP2=triton.next_power_of_2(self.E + 1),
        )
        return out


# Module-level shape shims rewritten by check.py / benchmark.py per shape.
T_total = 32768
H = 4096
I = 1536  # noqa: E41
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
