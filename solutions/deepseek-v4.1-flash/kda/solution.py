"""Kimi Delta Attention (KDA) forward, chunk form -- custom Triton implementation.

Math (matches reference.py / fla naive chunk KDA).  Per chunk of BT=64 tokens,
with gc the in-chunk cumsum of the log-decay g:

    kd = k*exp(gc)          ket = (k*exp(-gc))^T        qg = q*exp(gc)*scale
    Araw[c,i] = (kd @ ke^T)[c,i] = sum_d k[c,d] exp(gc[c,d]-gc[i,d]) k[i,d]
    M[c,i]    = beta[c]*Araw[c,i]   for c > i           (strictly lower)
    Tmat      = (I + M)^-1                              (unit lower triangular)
    w = Tmat @ (beta[:,None]*kd)      u = Tmat @ (beta[:,None]*v)

Inter-chunk recurrence (S_0 = 0, dec_i = exp(gc_i[BT-1])):

    v_new_i = u_i - w_i @ S_i
    o_i     = qg_i @ S_i + Aqk_i @ v_new_i,   Aqk_i[c,j] = qg_i[c].ket_i[:,j],  c >= j
    S_{i+1} = (S_i + ket_i @ v_new_i) * dec_i[:,None]

Kernel split (3 launches):
  A) parallel over every chunk: w, u, ket, qg, dec        (the "intra" part)
  B) sequential over chunks, parallel over (V/BV, B*H):   the only serial kernel,
     emits every S_i and v_new_i.  Loads only ket/w/u -- no cumsum, no fp32 g,
     no in-loop transposes.
  C) parallel over every (chunk, V-block): o              (the "output" part)

The triangular inverse uses the doubling identity
  (I - N)^-1 = (I+N)(I+N^2)(I+N^4)(I+N^8)(I+N^16)(I+N^32),  N = -M   (N^64 = 0)
The fp32 dots use input_precision="ieee": the stress cases use very large keys,
which makes |M| ~ 1e2 and the inverse badly conditioned for tf32.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

OP_TYPE = "linear_attention"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]


if triton is not None:

    # ------------------------------------------------------------------
    # Kernel A: intra-chunk -> w, u, ket, qg, dec
    # ------------------------------------------------------------------
    @triton.jit
    def _kda_intra_kernel(
        q_ptr, k_ptr, g_ptr, beta_ptr, v_ptr,
        w_ptr, u_ptr, ket_ptr, qg_ptr, dec_ptr,
        T, H, NT, scale,
        K: tl.constexpr, V: tl.constexpr,
        BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    ):
        i_n = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H

        o_t = i_n * BT + tl.arange(0, BT)
        o_b = tl.arange(0, BK)
        o_c = tl.arange(0, BT)[:, None]
        o_i = tl.arange(0, BT)[None, :]
        rows = tl.arange(0, BT)

        b_b = tl.load(beta_ptr + (i_b * T + o_t) * H + i_h).to(tl.float32)
        base = (i_b * T + o_t[:, None]) * H + i_h
        ket_base = ((i_b * H + i_h) * NT + i_n) * K

        # ---- Araw = kd @ ke^T (accumulate over K blocks); emit ket, qg, dec
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for kb in range(0, K, BK):
            off = base * K + (kb + o_b)[None, :]
            b_k = tl.load(k_ptr + off).to(tl.float32)
            b_g = tl.load(g_ptr + off)
            b_gc = tl.cumsum(b_g, axis=0)
            b_eg = tl.exp(b_gc)
            b_kd = (b_k * b_eg).to(tl.bfloat16)
            b_ke = (b_k * tl.exp(-b_gc)).to(tl.bfloat16)
            b_A = tl.dot(b_kd, tl.trans(b_ke), b_A)
            p_ket = ket_ptr + (ket_base + (kb + o_b)[None, :]) * BT + rows[:, None]
            tl.store(p_ket, b_ke)
            b_q = tl.load(q_ptr + off).to(tl.float32)
            tl.store(qg_ptr + off, (b_q * b_eg * scale).to(tl.bfloat16))
            b_last = tl.sum(tl.where((rows == BT - 1)[:, None], b_gc, 0.0), axis=0)
            p_dec = dec_ptr + ((i_b * NT + i_n) * H + i_h) * K + (kb + o_b)
            tl.store(p_dec, tl.exp(b_last))

        # ---- Tmat = (I + M)^-1, M strictly lower, row-scaled by beta
        b_M = tl.where(o_c > o_i, b_A * b_b[:, None], 0.0)
        b_N = -b_M
        b_I = tl.where(o_c == o_i, 1.0, 0.0)
        b_Tm = b_I + b_N
        b_P = b_N
        for _ in tl.static_range(5):
            b_P = tl.dot(b_P, b_P, input_precision="ieee")
            b_Tm = tl.dot(b_Tm, b_I + b_P, input_precision="ieee")

        # ---- w = Tmat @ (beta * exp(gc) * k)
        for kb in range(0, K, BK):
            off = base * K + (kb + o_b)[None, :]
            b_k = tl.load(k_ptr + off).to(tl.float32)
            b_g = tl.load(g_ptr + off)
            b_gc = tl.cumsum(b_g, axis=0)
            b_kd = b_k * tl.exp(b_gc)
            b_w = tl.dot(b_Tm, b_kd * b_b[:, None], input_precision="ieee")
            tl.store(w_ptr + off, b_w.to(tl.bfloat16))

        # ---- u = Tmat @ (beta * v)
        o_v = tl.arange(0, BV)
        for vb in range(0, V, BV):
            off = base * V + (vb + o_v)[None, :]
            b_v = tl.load(v_ptr + off).to(tl.float32)
            b_u = tl.dot(b_Tm, b_v * b_b[:, None], input_precision="ieee")
            tl.store(u_ptr + off, b_u.to(tl.bfloat16))

    # ------------------------------------------------------------------
    # Kernel B: sequential state pass -> all S_i (bf16) and v_new_i
    # ------------------------------------------------------------------
    @triton.jit
    def _kda_state_kernel(
        ket_ptr, w_ptr, u_ptr, dec_ptr, h_ptr, vn_ptr,
        T, H, NT,
        K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_bh = tl.program_id(1)
        i_b = i_bh // H
        i_h = i_bh % H

        o_k = tl.arange(0, K)
        o_t = tl.arange(0, BT)
        o_v = i_v * BV + tl.arange(0, BV)
        b_S = tl.zeros([K, BV], dtype=tl.float32)

        for i_n in range(0, NT):
            t_off = i_n * BT + o_t
            base = (i_b * T + t_off[:, None]) * H + i_h
            off_k = base * K + o_k[None, :]
            off_v = base * V + o_v[None, :]
            ket_base = ((i_b * H + i_h) * NT + i_n) * K

            b_ket = tl.load(ket_ptr + (ket_base + o_k[:, None]) * BT + o_t[None, :])
            b_w = tl.load(w_ptr + off_k)
            b_u = tl.load(u_ptr + off_v).to(tl.float32)

            p_h = h_ptr + (((i_b * NT + i_n) * H + i_h) * K + o_k[:, None]) * V + o_v[None, :]
            tl.store(p_h, b_S.to(tl.bfloat16))

            b_vn = b_u - tl.dot(b_w, b_S.to(tl.bfloat16))
            tl.store(vn_ptr + off_v, b_vn.to(tl.bfloat16))

            b_dec = tl.load(dec_ptr + ((i_b * NT + i_n) * H + i_h) * K + o_k)
            b_S = (b_S + tl.dot(b_ket, b_vn.to(tl.bfloat16))) * b_dec[:, None]

    # ------------------------------------------------------------------
    # Kernel C: fully parallel output pass -> o
    # ------------------------------------------------------------------
    @triton.jit
    def _kda_out_kernel(
        qg_ptr, ket_ptr, h_ptr, vn_ptr, o_ptr,
        T, H, NT,
        K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    ):
        i_vn = tl.program_id(0)
        i_bh = tl.program_id(1)
        NV = V // BV
        i_n = i_vn // NV
        i_v = i_vn % NV
        i_b = i_bh // H
        i_h = i_bh % H

        o_k = tl.arange(0, K)
        o_t = tl.arange(0, BT)
        o_v = i_v * BV + tl.arange(0, BV)
        o_c = tl.arange(0, BT)[:, None]
        o_j = tl.arange(0, BT)[None, :]

        t_off = i_n * BT + o_t
        base = (i_b * T + t_off[:, None]) * H + i_h
        off_k = base * K + o_k[None, :]
        off_v = base * V + o_v[None, :]
        ket_base = ((i_b * H + i_h) * NT + i_n) * K

        b_qg = tl.load(qg_ptr + off_k)
        b_ket = tl.load(ket_ptr + (ket_base + o_k[:, None]) * BT + o_t[None, :])
        b_Aqk = tl.dot(b_qg, b_ket)
        b_Aqk = tl.where(o_c >= o_j, b_Aqk, 0.0)

        p_h = h_ptr + (((i_b * NT + i_n) * H + i_h) * K + o_k[:, None]) * V + o_v[None, :]
        b_S = tl.load(p_h)
        b_vn = tl.load(vn_ptr + off_v)

        b_o = tl.dot(b_qg, b_S) + tl.dot(b_Aqk.to(tl.bfloat16), b_vn)
        tl.store(o_ptr + off_v, b_o.to(tl.bfloat16))


def _kda_forward(q, k, v, g, beta, scale, chunk_size=64):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    assert T % BT == 0 and K % 64 == 0 and V % 64 == 0

    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    g = g.contiguous(); beta = beta.contiguous()

    dev = q.device
    w = torch.empty(B, T, H, K, device=dev, dtype=torch.bfloat16)
    u = torch.empty(B, T, H, V, device=dev, dtype=torch.bfloat16)
    ket = torch.empty(B, H, NT, K, BT, device=dev, dtype=torch.bfloat16)
    qg = torch.empty(B, T, H, K, device=dev, dtype=torch.bfloat16)
    dec = torch.empty(B, NT, H, K, device=dev, dtype=torch.float32)
    h = torch.empty(B, NT, H, K, V, device=dev, dtype=torch.bfloat16)
    vn = torch.empty(B, T, H, V, device=dev, dtype=torch.bfloat16)
    o = torch.empty(B, T, H, V, device=dev, dtype=torch.bfloat16)

    BKA, BVA, BVB, BVC = 64, 64, 64, 64

    _kda_intra_kernel[(NT, B * H)](
        q, k, g, beta, v, w, u, ket, qg, dec,
        T, H, NT, scale,
        K=K, V=V, BT=BT, BK=BKA, BV=BVA,
        num_warps=8, num_stages=2,
    )
    _kda_state_kernel[(V // BVB, B * H)](
        ket, w, u, dec, h, vn,
        T, H, NT,
        K=K, V=V, BT=BT, BV=BVB,
        num_warps=8, num_stages=2,
    )
    _kda_out_kernel[(NT * (V // BVC), B * H)](
        qg, ket, h, vn, o,
        T, H, NT,
        K=K, V=V, BT=BT, BV=BVC,
        num_warps=4, num_stages=1,
    )
    return o


class Model(nn.Module):
    """KDA forward (chunk form). No learned parameters; all inputs are activations."""

    def __init__(self, B: int, T: int, H: int, K: int, V: int, chunk_size: int = 64):
        super().__init__()
        self.B, self.T, self.H, self.K, self.V = B, T, H, K, V
        self.chunk_size = chunk_size
        self.scale = float(K) ** -0.5
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, q, k, v, g, beta):
        return _kda_forward(q, k, v, g, beta, scale=self.scale, chunk_size=self.chunk_size)


# Module-level shape shims (overridden by check.py / benchmark.py per shape).
B = 2
T = 1024
H = 8
K = 128
V = 128
CHUNK_SIZE = 64


def get_inputs():
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16) * 0.1
    g = (torch.randn(B, T, H, K, dtype=torch.float32) * 0.1 - 0.05)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.bfloat16))
    return [q, k, v, g, beta]


def get_init_inputs():
    return [B, T, H, K, V, CHUNK_SIZE]
