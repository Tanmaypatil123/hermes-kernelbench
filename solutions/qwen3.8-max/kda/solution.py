"""Kimi Delta Attention forward (chunk form) -- custom Triton kernels for SM120.

Chunk-parallel decomposition (FLA-style math, independently implemented):
  1. intra kernel: per 64-token chunk, compute
       Aqk  = tril(scale * (q e^gc) (k e^-gc)^T)          (intra-chunk scores)
       Akk  = strict_tril(beta_row * (k e^gc)(k e^-gc)^T)  (delta-rule matrix)
       Tinv = (I + Akk)^{-1}   via repeated squaring in fp32 (exact for the
              nilpotent strictly-lower-triangular Akk; 5 squaring rounds cover
              all powers < 64).  fp32/ieee precision is REQUIRED here: the
              structured-key long-memory regime makes (I+Akk) ill-conditioned
              and tf32 blows up.
       w    = Tinv (beta * k * e^gc),  u = Tinv (beta * v)
       kg   = k * e^(g_last - gc),     g_last = chunk-total log2-decay
     Gate cumsum is computed inside the kernel in log2 domain; a mid-chunk
     anchor row keeps exp2 factors bounded.
  2. state kernel: sequential over chunks per (batch, head, V-block):
       v_new = u - w @ S ;  S <- e^(g_last) * S + kg^T @ v_new
     storing S (h) and v_new for the output kernel.
  3. output kernel: o = (scale * q e^gc) @ S + Aqk @ v_new.

All heavy lifting runs in tl.dot bf16 x bf16 -> fp32 except the triangular
inverse, which runs fp32 ieee. No torch tensor-core ops are used anywhere.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import triton
import triton.language as tl

RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit
def _kda_intra_kernel(
    q, k, v, g, beta,
    Aqk, w, u, kg, glast,
    scale, T, NT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    b_tok = (i_b * T + o_t) * H + i_h                       # (BT,)
    b_beta = tl.load(beta + b_tok).to(tl.float32)

    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Akk = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in tl.static_range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_gk = b_tok[:, None] * K + o_k[None, :]
        b_g = tl.load(g + p_gk)
        b_gc = tl.cumsum(b_g, axis=0) * RCP_LN2             # log2-domain cumsum
        b_gl = tl.sum(b_g, axis=0) * RCP_LN2                # (BK,) chunk total
        b_gn = tl.sum(tl.where((o_i == BT // 2)[:, None], b_gc, 0.0), axis=0)
        b_dq = tl.math.exp2(b_gc - b_gn[None, :])
        b_dk = tl.math.exp2(b_gn[None, :] - b_gc)
        b_k = tl.load(k + p_gk).to(tl.float32)
        b_q = tl.load(q + p_gk).to(tl.float32)
        b_kgt = tl.trans((b_k * b_dk).to(tl.bfloat16))      # (BK, BT)
        b_Aqk = tl.dot((b_q * b_dq).to(tl.bfloat16), b_kgt, b_Aqk)
        b_Akk = tl.dot((b_k * b_dq).to(tl.bfloat16), b_kgt, b_Akk)
        tl.store(glast + (i_bh * NT + i_t) * K + o_k, b_gl)
        tl.store(kg + p_gk, (b_k * tl.math.exp2(b_gl[None, :] - b_gc)).to(tl.bfloat16))

    b_Aqk = tl.where(o_i[:, None] >= o_i[None, :], b_Aqk * scale, 0.0)
    tl.store(Aqk + b_tok[:, None] * BT + o_i[None, :], b_Aqk.to(tl.bfloat16))
    b_Akk = tl.where(o_i[:, None] > o_i[None, :], b_Akk * b_beta[:, None], 0.0)

    # (I + Akk)^{-1} with Akk strictly lower triangular == sum_{p<BT} (-Akk)^p
    # = (I+S)(I+S^2)(I+S^4)(I+S^8)(I+S^16)(I+S^32),  S = -Akk
    b_S = -b_Akk
    b_P = b_S + (o_i[:, None] == o_i[None, :]).to(tl.float32)
    for _ in tl.static_range(5):
        b_S = tl.dot(b_S, b_S, input_precision="ieee")
        b_P = tl.dot(b_P, b_S, b_P, input_precision="ieee")
    b_P = tl.where(o_i[:, None] >= o_i[None, :], b_P, 0.0)
    b_T = b_P.to(tl.bfloat16)

    for i_k in tl.static_range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_gk = b_tok[:, None] * K + o_k[None, :]
        b_g = tl.load(g + p_gk)
        b_gc = tl.cumsum(b_g, axis=0) * RCP_LN2
        b_k = tl.load(k + p_gk).to(tl.float32)
        b_kw = (b_k * b_beta[:, None] * tl.math.exp2(b_gc)).to(tl.bfloat16)
        b_w = tl.dot(b_T, b_kw)
        tl.store(w + p_gk, b_w.to(tl.bfloat16))

    for i_v in tl.static_range(V // 64):
        o_v = i_v * 64 + tl.arange(0, 64)
        p_v = b_tok[:, None] * V + o_v[None, :]
        b_v = tl.load(v + p_v).to(tl.float32)
        b_uv = (b_v * b_beta[:, None]).to(tl.bfloat16)
        b_u = tl.dot(b_T, b_uv)
        tl.store(u + p_v, b_u.to(tl.bfloat16))


@triton.jit
def _kda_h_kernel(
    w, u, kg, glast, h, v_new,
    NT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
):
    # NOTE: num_warps must stay 2 on Blackwell -- a Triton tl.dot
    # accumulator-recurrence race affects wider configs (FLA found the same).
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    T = NT * BT
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    o_t = tl.arange(0, BT)

    b_h = tl.zeros([K, BV], dtype=tl.float32)
    for i_t in range(NT):
        p_h = h + ((i_bh * NT + i_t) * K + o_k[:, None]) * V + o_v[None, :]
        tl.store(p_h, b_h.to(tl.bfloat16))

        b_tok = (i_b * T + i_t * BT + o_t) * H + i_h
        b_w = tl.load(w + b_tok[:, None] * K + o_k[None, :])
        b_v = tl.dot(b_w, b_h.to(tl.bfloat16))               # (BT, BV) fp32
        b_u = tl.load(u + b_tok[:, None] * V + o_v[None, :]).to(tl.float32)
        b_vn = b_u - b_v
        tl.store(v_new + b_tok[:, None] * V + o_v[None, :], b_vn.to(tl.bfloat16))

        b_gl = tl.load(glast + (i_bh * NT + i_t) * K + o_k)
        b_h = b_h * tl.math.exp2(b_gl)[:, None]
        b_kgT = tl.load(kg + b_tok[None, :] * K + o_k[:, None])   # (K, BT)
        b_h = tl.dot(b_kgT, b_vn.to(tl.bfloat16), b_h)


@triton.jit
def _kda_o_kernel(
    q, g, Aqk, h, v_new, o,
    scale, NT,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    T = NT * BT
    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    b_tok = (i_b * T + o_t) * H + i_h
    o_v = tl.arange(0, V)

    b_o = tl.zeros([BT, V], dtype=tl.float32)
    for i_k in tl.static_range(K // BK):
        o_k = i_k * BK + tl.arange(0, BK)
        p_qk = b_tok[:, None] * K + o_k[None, :]
        b_q = tl.load(q + p_qk).to(tl.float32)
        b_g = tl.load(g + p_qk)
        b_gc = tl.cumsum(b_g, axis=0) * RCP_LN2
        b_qg = (b_q * tl.math.exp2(b_gc) * scale).to(tl.bfloat16)
        p_h = h + ((i_bh * NT + i_t) * K + o_k[:, None]) * V + o_v[None, :]
        b_h = tl.load(p_h)
        b_o = tl.dot(b_qg, b_h, b_o)

    b_A = tl.load(Aqk + b_tok[:, None] * BT + o_i[None, :])
    b_vn = tl.load(v_new + b_tok[:, None] * V + o_v[None, :])
    b_o = tl.dot(b_A, b_vn, b_o)
    tl.store(o + b_tok[:, None] * V + o_v[None, :], b_o.to(tl.bfloat16))


_PROF_COUNT: dict = {}


def _forward_impl(q, k, v, g, beta, scale, BT, prof):
    B, T, H, K = q.shape
    V = v.shape[-1]
    NT = T // BT
    dev = q.device

    Aqk = torch.empty(B, T, H, BT, device=dev, dtype=torch.bfloat16)
    w = torch.empty(B, T, H, K, device=dev, dtype=torch.bfloat16)
    u = torch.empty(B, T, H, V, device=dev, dtype=torch.bfloat16)
    kg = torch.empty(B, T, H, K, device=dev, dtype=torch.bfloat16)
    glast = torch.empty(B, H, NT, K, device=dev, dtype=torch.float32)
    h = torch.empty(B, H, NT, K, V, device=dev, dtype=torch.bfloat16)
    v_new = torch.empty(B, T, H, V, device=dev, dtype=torch.bfloat16)
    o = torch.empty(B, T, H, V, device=dev, dtype=torch.bfloat16)

    BK = 32
    BV = 64
    grid_tc = (NT, B * H)

    def _launch():
        _kda_intra_kernel[grid_tc](
            q, k, v, g, beta, Aqk, w, u, kg, glast,
            scale, T, NT,
            H=H, K=K, V=V, BT=BT, BK=BK,
            num_warps=8, num_stages=2,
        )
        _kda_h_kernel[(V // BV, B * H)](
            w, u, kg, glast, h, v_new, NT,
            H=H, K=K, V=V, BT=BT, BV=BV,
            num_warps=2, num_stages=3,
        )
        _kda_o_kernel[grid_tc](
            q, g, Aqk, h, v_new, o,
            scale, NT,
            H=H, K=K, V=V, BT=BT, BK=BK,
            num_warps=8, num_stages=2,
        )

    if not prof:
        _launch()
        return o

    ts = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _kda_intra_kernel[grid_tc](
        q, k, v, g, beta, Aqk, w, u, kg, glast,
        scale, T, NT, H=H, K=K, V=V, BT=BT, BK=BK,
        num_warps=8, num_stages=2,
    )
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    _kda_h_kernel[(V // BV, B * H)](
        w, u, kg, glast, h, v_new, NT,
        H=H, K=K, V=V, BT=BT, BV=BV, num_warps=2, num_stages=3,
    )
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    _kda_o_kernel[grid_tc](
        q, g, Aqk, h, v_new, o,
        scale, NT, H=H, K=K, V=V, BT=BT, BK=BK,
        num_warps=8, num_stages=2,
    )
    torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
    print(f"PROF B={B} T={T} H={H} intra={ts[0]*1e3:.3f}ms h={ts[1]*1e3:.3f}ms o={ts[2]*1e3:.3f}ms", flush=True)
    return o


class Model(nn.Module):
    """KDA forward (chunk form). No learned parameters; all inputs are activations."""

    def __init__(self, B: int, T: int, H: int, K: int, V: int, chunk_size: int = 64):
        super().__init__()
        self.B, self.T, self.H, self.K, self.V = B, T, H, K, V
        self.chunk_size = chunk_size
        self.scale = float(K) ** -0.5
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        assert self.chunk_size == 64 and q.shape[-1] == 128 and v.shape[-1] == 128
        key = (q.shape[0], q.shape[1], q.shape[2])
        n = _PROF_COUNT.get(key, 0)
        _PROF_COUNT[key] = n + 1
        prof = n < 12
        return _forward_impl(q, k, v, g, beta, self.scale, self.chunk_size, prof)


# Module-level shape shims (overridden by check.py / benchmark.py per shape).
B = 2
T = 1024
H = 8
K = 128
V = 128
CHUNK_SIZE = 64


def get_inputs():
    """Return a list of activations for one forward call."""
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16) * 0.1
    g = (torch.randn(B, T, H, K, dtype=torch.float32) * 0.1 - 0.05)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.bfloat16))
    return [q, k, v, g, beta]


def get_init_inputs():
    return [B, T, H, K, V, CHUNK_SIZE]
