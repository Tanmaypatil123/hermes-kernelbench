"""Kimi Delta Attention (chunk forward) — custom Triton implementation.

Three-stage chunk-parallel pipeline (no FLA calls):
  Stage A (parallel over B*H*NT): in-chunk cumsum of g, the strictly-lower
    K-K interaction matrix, (I-A)^-1 via blocked forward substitution +
    Neumann combining, the WY representation w/u, the intra-chunk qk
    attention matrix Aqk, and decayed factors qg/kg reused later.
  Stage B (sequential over chunks, parallel over B*H x V-blocks): the
    inter-chunk state recurrence S, storing per-chunk states h and the
    corrected values v_i = u - w @ S. Loads are software double-buffered.
  Stage C (parallel over B*H*NT x V-blocks): o = (q*exp(G)) @ S + Aqk @ v_i.

Buffers are cached per shape to avoid allocation overhead in the timed loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Stage A: intra-chunk precompute
# --------------------------------------------------------------------------
@triton.jit
def _kda_intra_kernel(
    q, k, v, g, beta,
    w, u, qg, kg, aqk, glast,
    scale,
    T, H, K, V, NT,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    pid = tl.program_id(0)
    i_n = pid % NT
    i_bh = pid // NT
    i_b = i_bh // H
    i_h = i_bh % H

    rows = tl.arange(0, BT)
    dk = tl.arange(0, BK)
    dv = tl.arange(0, BV)

    base_k = ((i_b * T + i_n * BT) * H + i_h) * K
    base_v = ((i_b * T + i_n * BT) * H + i_h) * V
    HK = H * K
    HV = H * V

    b_g = tl.load(g + base_k + rows[:, None] * HK + dk[None, :])
    G = tl.cumsum(b_g, axis=0)  # (BT, BK) fp32

    b_k = tl.load(k + base_k + rows[:, None] * HK + dk[None, :]).to(tl.float32)
    b_q = tl.load(q + base_k + rows[:, None] * HK + dk[None, :]).to(tl.float32) * scale
    b_v = tl.load(v + base_v + rows[:, None] * HV + dv[None, :]).to(tl.float32)
    b_beta = tl.load(beta + (i_b * T + i_n * BT) * H + i_h + rows * H).to(tl.float32)

    # chunk-mid row and last row of G
    c = tl.sum(tl.where((rows == BT // 2)[:, None], G, 0.0), axis=0)      # (BK,)
    gl = tl.sum(tl.where((rows == BT - 1)[:, None], G, 0.0), axis=0)      # (BK,)

    LOG2E: tl.constexpr = 1.4426950408889634
    E = tl.math.exp2((G - c[None, :]) * LOG2E)
    Ei = tl.math.exp2((c[None, :] - G) * LOG2E)
    ec = tl.math.exp2(c * LOG2E)          # (BK,)
    eglc = tl.math.exp2((gl - c) * LOG2E)  # (BK,)

    b_pk = b_k * E
    b_p = b_q * E
    b_r = b_k * Ei
    b_qg = b_p * ec[None, :]               # q * exp(G) = (q*E) * exp(c)
    b_kg = b_r * eglc[None, :]             # k * exp(gl - G) = (k*Ei) * exp(gl-c)
    b_keg = b_pk * ec[None, :]             # k * exp(G)

    # intra-chunk qk attention (kept: diagonal and below)
    b_pb = b_p.to(tl.bfloat16)
    b_rb = b_r.to(tl.bfloat16)
    b_Aqk = tl.dot(b_pb, tl.trans(b_rb))
    b_Aqk = tl.where(rows[:, None] >= rows[None, :], b_Aqk, 0.0)

    # strictly-lower K-K interaction: A[i,j] = beta_i * sum_d k_i k_j exp(G_i - G_j)
    b_A = tl.dot(b_pk.to(tl.bfloat16), tl.trans(b_rb))
    b_A = tl.where(rows[:, None] > rows[None, :], b_A, 0.0)
    b_A = -b_A * b_beta[:, None]

    # Blocked (I - A)^-1.
    # Diagonal 16x16 blocks: Xd = (I+Ad)(I+Ad^2)(I+Ad^4)(I+Ad^8)  (Ad nilpotent)
    SB: tl.constexpr = 16
    blk = rows // SB
    same_blk = blk[:, None] == blk[None, :]
    low = rows[:, None] > rows[None, :]
    eye = tl.where(rows[:, None] == rows[None, :], 1.0, 0.0)
    b_Ad = tl.where(same_blk & low, b_A, 0.0).to(tl.bfloat16)
    b_Ad2 = tl.dot(b_Ad, b_Ad).to(tl.bfloat16)
    b_Ad4 = tl.dot(b_Ad2, b_Ad2).to(tl.bfloat16)
    b_Ad8 = tl.dot(b_Ad4, b_Ad4).to(tl.bfloat16)
    b_Ad3 = tl.dot(b_Ad, b_Ad2)
    b_Q = (eye + b_Ad + b_Ad3 + b_Ad2).to(tl.bfloat16)   # (I+Ad)(I+Ad^2)
    b_R = (b_Q + tl.dot(b_Q, b_Ad4)).to(tl.bfloat16)
    b_Xd = (b_R + tl.dot(b_R, b_Ad8)).to(tl.bfloat16)
    # Off-diagonal block levels: X = (I - N)(I + N^2) Xd, N = Xd L
    b_L = (-(b_A - tl.where(same_blk & low, b_A, 0.0))).to(tl.bfloat16)
    b_N = tl.dot(b_Xd, b_L).to(tl.bfloat16)
    b_t = (b_Xd + tl.dot(tl.dot(b_N, b_N).to(tl.bfloat16), b_Xd)).to(tl.bfloat16)
    b_X = (b_t - tl.dot(b_N, b_t)).to(tl.bfloat16)

    b_w = tl.dot(b_X, (b_keg * b_beta[:, None]).to(tl.bfloat16))   # (BT, BK)
    b_u = tl.dot(b_X, (b_v * b_beta[:, None]).to(tl.bfloat16))     # (BT, BV)

    tl.store(w + base_k + rows[:, None] * HK + dk[None, :], b_w.to(w.dtype.element_ty))
    tl.store(u + base_v + rows[:, None] * HV + dv[None, :], b_u.to(u.dtype.element_ty))
    tl.store(qg + base_k + rows[:, None] * HK + dk[None, :], b_qg.to(qg.dtype.element_ty))
    tl.store(kg + base_k + rows[:, None] * HK + dk[None, :], b_kg.to(kg.dtype.element_ty))
    abase = ((i_bh * NT + i_n) * BT) * BT
    tl.store(aqk + abase + rows[:, None] * BT + rows[None, :], b_Aqk.to(aqk.dtype.element_ty))
    tl.store(glast + (i_bh * NT + i_n) * K + dk, gl)


# --------------------------------------------------------------------------
# Stage B: inter-chunk state recurrence (sequential over NT, double-buffered)
# --------------------------------------------------------------------------
@triton.jit
def _kda_state_kernel(
    w, u, kg, glast,
    h, v_new,
    T, H, K, V, NT,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    rows = tl.arange(0, BT)
    dk = tl.arange(0, BK)
    dv = i_v * BV + tl.arange(0, BV)

    HK = H * K
    HV = H * V

    base_k = ((i_b * T) * H + i_h) * K
    base_v = ((i_b * T) * H + i_h) * V

    b_w = tl.load(w + base_k + rows[:, None] * HK + dk[None, :])
    b_u = tl.load(u + base_v + rows[:, None] * HV + dv[None, :]).to(tl.float32)
    b_kg = tl.load(kg + base_k + rows[:, None] * HK + dk[None, :])
    b_gl = tl.load(glast + i_bh * NT * K + dk)

    b_S = tl.zeros((BK, BV), dtype=tl.float32)

    for i_n in range(NT):
        # prefetch next chunk's tiles while computing the current one
        b_w_n = tl.load(w + base_k + (i_n + 1) * BT * HK + rows[:, None] * HK + dk[None, :],
                        mask=(i_n + 1 < NT) & (rows[:, None] >= 0), other=0.0)
        b_u_n = tl.load(u + base_v + (i_n + 1) * BT * HV + rows[:, None] * HV + dv[None, :],
                        mask=(i_n + 1 < NT) & (rows[:, None] >= 0), other=0.0).to(tl.float32)
        b_kg_n = tl.load(kg + base_k + (i_n + 1) * BT * HK + rows[:, None] * HK + dk[None, :],
                         mask=(i_n + 1 < NT) & (rows[:, None] >= 0), other=0.0)
        b_gl_n = tl.load(glast + (i_bh * NT + i_n + 1) * K + dk,
                         mask=(i_n + 1 < NT) & (dk >= 0), other=0.0)

        b_vi = b_u - tl.dot(b_w, b_S.to(b_w.dtype))

        tl.store(v_new + base_v + i_n * BT * HV + rows[:, None] * HV + dv[None, :],
                 b_vi.to(v_new.dtype.element_ty))
        h_base = ((i_bh * NT + i_n) * K) * V
        tl.store(h + h_base + dk[:, None] * V + dv[None, :], b_S.to(h.dtype.element_ty))

        b_S = b_S * tl.math.exp2(b_gl * 1.4426950408889634)[:, None] + tl.dot(tl.trans(b_kg), b_vi.to(b_kg.dtype))

        b_w = b_w_n
        b_u = b_u_n
        b_kg = b_kg_n
        b_gl = b_gl_n


# --------------------------------------------------------------------------
# Stage C: output
# --------------------------------------------------------------------------
@triton.jit
def _kda_output_kernel(
    qg, aqk, h, v_new,
    o,
    T, H, K, V, NT,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    pid = tl.program_id(0)
    i_v = tl.program_id(1)
    i_n = pid % NT
    i_bh = pid // NT
    i_b = i_bh // H
    i_h = i_bh % H

    rows = tl.arange(0, BT)
    dk = tl.arange(0, BK)
    dv = i_v * BV + tl.arange(0, BV)

    HK = H * K
    HV = H * V
    base_k = ((i_b * T + i_n * BT) * H + i_h) * K
    base_v = ((i_b * T + i_n * BT) * H + i_h) * V

    b_qg = tl.load(qg + base_k + rows[:, None] * HK + dk[None, :])
    abase = ((i_bh * NT + i_n) * BT) * BT
    b_Aqk = tl.load(aqk + abase + rows[:, None] * BT + rows[None, :])
    b_vi = tl.load(v_new + base_v + rows[:, None] * HV + dv[None, :])

    h_base = ((i_bh * NT + i_n) * K) * V
    b_h = tl.load(h + h_base + dk[:, None] * V + dv[None, :])

    b_o = tl.dot(b_qg, b_h) + tl.dot(b_Aqk, b_vi)

    tl.store(o + base_v + rows[:, None] * HV + dv[None, :],
             b_o.to(o.dtype.element_ty))


# --------------------------------------------------------------------------
# Host wrapper
# --------------------------------------------------------------------------
_BUFS: dict = {}
_GRAPHS: dict = {}


def _kda_chunk_fwd_impl(q, k, v, g, beta, scale, chunk_size, buf):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    BK = K
    BV = 32 if V >= 32 else V

    w, u, qg, kg = buf["w"], buf["u"], buf["qg"], buf["kg"]
    aqk, glast, h, v_new = buf["aqk"], buf["glast"], buf["h"], buf["v_new"]
    o = torch.empty(B, T, H, V, dtype=torch.bfloat16, device=q.device)

    _kda_intra_kernel[(B * H * NT,)](
        q, k, v, g, beta, w, u, qg, kg, aqk, glast,
        scale, T, H, K, V, NT,
        BT=BT, BK=BK, BV=V, num_warps=4,
    )
    _kda_state_kernel[(B * H, V // BV)](
        w, u, kg, glast, h, v_new,
        T, H, K, V, NT,
        BT=BT, BK=BK, BV=BV, num_warps=4,
    )
    _kda_output_kernel[(B * H * NT, V // 64)](
        qg, aqk, h, v_new, o,
        T, H, K, V, NT,
        BT=BT, BK=BK, BV=64, num_warps=4,
    )
    return o


def _make_bufs(B, T, H, K, V, NT, BT, dev):
    return {
        "w": torch.empty(B, T, H, K, dtype=torch.bfloat16, device=dev),
        "u": torch.empty(B, T, H, V, dtype=torch.bfloat16, device=dev),
        "qg": torch.empty(B, T, H, K, dtype=torch.bfloat16, device=dev),
        "kg": torch.empty(B, T, H, K, dtype=torch.bfloat16, device=dev),
        "aqk": torch.empty(B * H * NT, BT, BT, dtype=torch.bfloat16, device=dev),
        "glast": torch.empty(B * H * NT, K, dtype=torch.float32, device=dev),
        "h": torch.empty(B * H * NT, K, V, dtype=torch.bfloat16, device=dev),
        "v_new": torch.empty(B, T, H, V, dtype=torch.bfloat16, device=dev),
    }


def _kda_chunk_fwd(q, k, v, g, beta, scale: float, chunk_size: int = 64):
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    dev = q.device
    key = (B, T, H, K, V, dev.index)

    ent = _GRAPHS.get(key)
    if ent is None:
        buf = _make_bufs(B, T, H, K, V, NT, BT, dev)
        ent = {"buf": buf, "graph": None}
        _GRAPHS[key] = ent
        # first call: eager (also compiles the kernels)
        return _kda_chunk_fwd_impl(q, k, v, g, beta, scale, chunk_size, buf)

    if ent["graph"] is None:
        # second call: try to capture a CUDA graph on static buffers
        try:
            sq = torch.empty_like(q); sk = torch.empty_like(k)
            sv = torch.empty_like(v); sg = torch.empty_like(g)
            sb = torch.empty_like(beta)
            sq.copy_(q); sk.copy_(k); sv.copy_(v); sg.copy_(g); sb.copy_(beta)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                _kda_chunk_fwd_impl(sq, sk, sv, sg, sb, scale, chunk_size, ent["buf"])
            torch.cuda.current_stream().wait_stream(s)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                o_s = _kda_chunk_fwd_impl(sq, sk, sv, sg, sb, scale, chunk_size, ent["buf"])
            ent.update(graph=gr, ins=(sq, sk, sv, sg, sb), out=o_s)
        except Exception:
            ent["graph"] = False
            return _kda_chunk_fwd_impl(q, k, v, g, beta, scale, chunk_size, ent["buf"])

    if ent["graph"]:
        sq, sk, sv, sg, sb = ent["ins"]
        sq.copy_(q); sk.copy_(k); sv.copy_(v); sg.copy_(g); sb.copy_(beta)
        ent["graph"].replay()
        return ent["out"].clone()
    return _kda_chunk_fwd_impl(q, k, v, g, beta, scale, chunk_size, ent["buf"])


class Model(nn.Module):
    def __init__(self, B: int, T: int, H: int, K: int, V: int, chunk_size: int = 64):
        super().__init__()
        self.B, self.T, self.H, self.K, self.V = B, T, H, K, V
        self.chunk_size = chunk_size
        self.scale = float(K) ** -0.5
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)

    def forward(self, q, k, v, g, beta):
        return _kda_chunk_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(),
            g.contiguous(), beta.contiguous(),
            scale=self.scale, chunk_size=self.chunk_size,
        )


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
