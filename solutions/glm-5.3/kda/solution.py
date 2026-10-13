"""Kimi Delta Attention (KDA) forward — chunk-parallel form, custom Triton kernels for
SM120 (RTX PRO 6000 / Blackwell).

Written from scratch (no FLA imports). Math validated against the naive reference:

  per chunk of BT=64 rows (q,k: (B,T,H,K) bf16, v: (B,T,H,V) bf16, g: fp32 log-decay,
  beta: (B,T,H) bf16, scale float):

    gc   = cumsum_inclusive(g, rows)                    # in-chunk decay cumsum
    kk[c,i] = sum_d k[c,d] * exp(gc[c,d]-gc[i,d]) * k[i,d]        (c > i)
    M    = beta_c * kk  (strictly lower)
    P    = (I + M)^-1      via Neumann doubling (M nilpotent, index <= BT)
    w    = P @ (beta_row * exp(gc) * k)      (= A_final @ (exp(gc)*k))
    u    = P @ (beta_row * v)                 (= A_final @ v)
    Aqk[c,j] = sum_d q[c,d] * exp(gc[c,d]-gc[j,d]) * k[j,d]   (keep c >= j)
    z    = q*scale*exp(gc) - Aqk @ w
    t2   = Aqk @ u
    kg2[c,d] = k[c,d] * exp(gc_last[d] - gc[c,d])
    d    = exp(gc_last)

  sequential inter-chunk pass (state S, (K,V) fp32, S_0 = 0):
    v_i = u - w @ S
    o   = z @ S + t2              # == (q*exp(gc)) @ S + Aqk @ v_i
    S   = S * d[:,None] + kg2^T @ v_i

Kernel 1 (`_kda_prep_kernel`) does everything chunk-local, fully parallel over
B*H*NT programs. Kernel 2 (`_kda_seq_kernel`) walks the NT chunks sequentially,
parallel over (B*H) x V-blocks, with the state tile resident in registers.
In the seq kernel w and z are loaded as one stacked (2*BT, K) tile C = [w; z]
so w@S and z@S share a single dot (C@S), and kg2 is loaded padded to
(2*BT, K) with zero rows so the same C-style shapes feed the state update.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

OP_TYPE = "linear_attention"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]


@triton.jit
def _kda_prep_kernel(
    Q, K, V, G, BETA,
    W, Z, U, T2, KG2, Dc,
    scale,
    T, H, NT,
    DK: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid % NT
    bh = pid // NT
    h = bh % H
    b = bh // H

    r_c = tl.arange(0, BT)          # rows within chunk
    r_d = tl.arange(0, DK)          # full K head dim (128)
    r_v = tl.arange(0, DV)          # full V head dim (128)

    # ---- load g, chunk-local inclusive cumsum ----
    g_ptrs = G + ((b * T + n * BT + r_c[:, None]) * H + h) * DK + r_d[None, :]
    gt = tl.load(g_ptrs)                        # (BT, DK) fp32
    gc = tl.cumsum(gt, axis=0)                 # inclusive cumsum over rows
    m = tl.min(gc, axis=0)                     # (DK,) per-channel min
    lastv = tl.min(tl.where(r_c[:, None] == BT - 1, gc, float("inf")), axis=0)  # (DK,)
    e_gc = tl.exp(gc)

    beta_c = tl.load(BETA + (b * T + n * BT + r_c) * H + h).to(tl.float32)  # (BT,)

    k_ptrs = K + ((b * T + n * BT + r_c[:, None]) * H + h) * DK + r_d[None, :]
    kt = tl.load(k_ptrs).to(tl.float32)        # (BT, DK)

    # ---- decay-scaled k.k^T -> strictly-lower M, then (I+M)^-1 by doubling ----
    kgp = (kt * tl.exp(gc - m[None, :])).to(tl.bfloat16)
    kgm = (kt * tl.exp(m[None, :] - gc)).to(tl.bfloat16)
    kk = tl.dot(kgp, tl.trans(kgm))            # (BT, BT) fp32 accum
    lower = (r_c[:, None] > r_c[None, :]).to(tl.float32)
    M = kk * beta_c[:, None] * lower
    Wm = -M
    eye = tl.where(r_c[:, None] == r_c[None, :], 1.0, 0.0)
    P = eye + Wm                               # (I + W), W = -M
    # NOTE: the alternating Neumann series suffers deep cancellation in the
    # structured (beta~0.8, near-uniform k) regime -- tf32 is NOT enough here;
    # these dots must run in fp32 (ieee).
    X = tl.dot(Wm, Wm, input_precision="ieee")
    for _ in tl.static_range(5):
        P = P + tl.dot(P, X, input_precision="ieee")
        X = tl.dot(X, X, input_precision="ieee")
    # P = prod_{k<=5} (I + W^(2^k)) = sum_j (-M)^j = (I + M)^-1

    # ---- w = P @ (beta_row * exp(gc) * k);  u = P @ (beta_row * v) ----
    kv = kt * e_gc * beta_c[:, None]           # (BT, DK) fp32
    w = tl.dot(P, kv, input_precision="tf32")  # (BT, DK)
    v_ptrs = V + ((b * T + n * BT + r_c[:, None]) * H + h) * DV + r_v[None, :]
    vv = tl.load(v_ptrs).to(tl.float32) * beta_c[:, None]
    u = tl.dot(P, vv, input_precision="tf32")  # (BT, DV)

    w_b = w.to(tl.bfloat16)
    u_b = u.to(tl.bfloat16)

    # ---- Aqk (keep c >= j), then z, t2 ----
    q_ptrs = Q + ((b * T + n * BT + r_c[:, None]) * H + h) * DK + r_d[None, :]
    qt = tl.load(q_ptrs).to(tl.float32) * scale
    qgp = (qt * tl.exp(gc - m[None, :])).to(tl.bfloat16)
    Aqk = tl.dot(qgp, tl.trans(kgm))           # (BT, BT) fp32 accum
    keep = (r_c[:, None] >= r_c[None, :]).to(tl.float32)
    Aqk = Aqk * keep
    Aqk_b = Aqk.to(tl.bfloat16)
    qg = qt * e_gc                             # (BT, DK) fp32
    z = qg - tl.dot(Aqk_b, w_b)
    t2 = tl.dot(Aqk_b, u_b)

    # ---- kg2 and store ----
    kg2 = (kt * tl.exp(lastv[None, :] - gc)).to(tl.bfloat16)
    d_last = tl.exp(lastv)

    out_row = ((bh * NT + n) * BT + r_c[:, None])
    tl.store(W + out_row * DK + r_d[None, :], w_b)
    tl.store(Z + out_row * DK + r_d[None, :], z.to(tl.bfloat16))
    tl.store(U + out_row * DV + r_v[None, :], u_b)
    tl.store(T2 + out_row * DV + r_v[None, :], t2.to(tl.bfloat16))
    tl.store(KG2 + out_row * DK + r_d[None, :], kg2)
    tl.store(Dc + (bh * NT + n) * DK + r_d, d_last)


@triton.jit
def _kda_seq_kernel(
    W, Z, U, T2, KG2, Dc, O,
    H, NT,
    T,
    DK: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
):
    pid = tl.program_id(0)
    nv = DV // BV
    bh = pid // nv
    v0 = pid % nv

    r2 = tl.arange(0, 2 * BT)          # stacked rows: [0,BT) = w/z rows, [BT,2BT) padding
    r_c = tl.arange(0, BT)
    r_d = tl.arange(0, DK)
    r_v = v0 * BV + tl.arange(0, BV)

    S = tl.zeros((DK, BV), dtype=tl.float32)

    for n in range(NT):
        row2 = ((bh * NT + n) * BT + (r2[:, None] % BT))
        # C = [w; z] stacked (2*BT, DK) bf16
        cw = tl.load(W + row2 * DK + r_d[None, :], mask=r2[:, None] < BT, other=0.0)
        cz = tl.load(Z + row2 * DK + r_d[None, :], mask=r2[:, None] >= BT, other=0.0)
        C = cw + cz
        # kg2 padded to (2*BT, DK): rows >= BT are zero
        kg2p = tl.load(KG2 + row2 * DK + r_d[None, :], mask=r2[:, None] < BT, other=0.0)
        # u at stacked rows < BT, zeros elsewhere
        ut = tl.load(U + row2 * DV + r_v[None, :], mask=r2[:, None] < BT, other=0.0).to(tl.float32)
        # t2 at stacked rows >= BT (i.e. chunk rows shifted), zeros elsewhere
        t2t = tl.load(T2 + row2 * DV + r_v[None, :], mask=r2[:, None] >= BT, other=0.0).to(tl.float32)
        d = tl.load(Dc + (bh * NT + n) * DK + r_d)            # (DK,)

        Sb = S.to(tl.bfloat16)
        CS = tl.dot(C, Sb)                                     # (2*BT, BV) = [w@S; z@S]
        v_full = ut - CS                                      # rows <BT: v_i ; rows >=BT: -z@S
        o_full = CS + t2t                                     # rows >=BT: z@S + t2 = o
        S = S * d[:, None] + tl.dot(tl.trans(kg2p), v_full.to(tl.bfloat16))

        # store o (rows >= BT)
        h = bh % H
        b = bh // H
        o_mask = r2[:, None] >= BT
        o_ptrs = O + ((b * T + n * BT + (r2[:, None] - BT)) * H + h) * DV + r_v[None, :]
        tl.store(o_ptrs, o_full.to(tl.bfloat16), mask=o_mask)


@triton.jit
def _kda_seq_v2_kernel(
    W, Z, U, T2, KG2, Dc, O,
    H, NT,
    T,
    DK: tl.constexpr, DV: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
):
    """Fallback seq kernel: separate w/z dots, no stacked tiles (smaller smem)."""
    pid = tl.program_id(0)
    nv = DV // BV
    bh = pid // nv
    v0 = pid % nv
    r_c = tl.arange(0, BT)
    r_d = tl.arange(0, DK)
    r_v = v0 * BV + tl.arange(0, BV)
    S = tl.zeros((DK, BV), dtype=tl.float32)
    for n in range(NT):
        row = ((bh * NT + n) * BT + r_c[:, None])
        wt = tl.load(W + row * DK + r_d[None, :])
        zt = tl.load(Z + row * DK + r_d[None, :])
        kg2 = tl.load(KG2 + row * DK + r_d[None, :])
        ut = tl.load(U + row * DV + r_v[None, :]).to(tl.float32)
        t2t = tl.load(T2 + row * DV + r_v[None, :]).to(tl.float32)
        d = tl.load(Dc + (bh * NT + n) * DK + r_d)
        Sb = S.to(tl.bfloat16)
        wS = tl.dot(wt, Sb)
        v_i = ut - wS
        zS = tl.dot(zt, Sb)
        o = zS + t2t
        S = S * d[:, None] + tl.dot(tl.trans(kg2), v_i.to(tl.bfloat16))
        h = bh % H
        b = bh // H
        o_ptrs = O + ((b * T + n * BT + r_c[:, None]) * H + h) * DV + r_v[None, :]
        tl.store(o_ptrs, o.to(tl.bfloat16))


_PROFILED = {"done": False}
_SEQ_OK = {"merged": None}
_BUFS = {}


def _get_bufs(B, T, H, DK, DV, BT, NT, dev):
    key = (B, T, H, DK, DV, BT, dev)
    bufs = _BUFS.get(key)
    NPROG = B * H * NT
    if bufs is None or bufs[0].numel() != NPROG * BT:
        W = torch.empty((NPROG * BT, DK), dtype=torch.bfloat16, device=dev)
        bufs = (W, torch.empty_like(W), torch.empty_like(W),
                torch.empty((NPROG * BT, DV), dtype=torch.bfloat16, device=dev), None,
                torch.empty((NPROG, DK), dtype=torch.float32, device=dev))
        bufs = list(bufs)
        bufs[4] = torch.empty_like(bufs[3])
        _BUFS[key] = bufs
    return bufs


def kda_forward(q, k, v, g, beta, scale: float, chunk_size: int = 64) -> torch.Tensor:
    B, T, H, DK = q.shape
    DV = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    assert T % BT == 0
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    dev = q.device
    NPROG = B * H * NT

    W, Z, KG2, U, T2, Dc = _get_bufs(B, T, H, DK, DV, BT, NT, dev)
    o = torch.empty((B, T, H, DV), dtype=torch.bfloat16, device=dev)

    def _prep():
        _kda_prep_kernel[(NPROG,)](
            q, k, v, g, beta,
            W, Z, U, T2, KG2, Dc,
            scale, T, H, NT,
            DK=DK, DV=DV, BT=BT,
            num_warps=8,
        )

    BV = 32

    def _seq():
        grid = (B * H * (DV // BV),)
        args = (W, Z, U, T2, KG2, Dc, o, H, NT, T)
        kw = dict(DK=DK, DV=DV, BT=BT, BV=BV, num_warps=8)
        if _SEQ_OK["merged"] is None:
            try:
                _kda_seq_kernel[grid](*args, **kw, num_stages=1)
                _SEQ_OK["merged"] = True
                return
            except Exception as exc:
                print(f"KDA_SEQ: merged kernel failed ({type(exc).__name__}); using fallback", flush=True)
                _SEQ_OK["merged"] = False
        if _SEQ_OK["merged"]:
            _kda_seq_kernel[grid](*args, **kw, num_stages=1)
        else:
            _kda_seq_v2_kernel[grid](*args, **kw)

    if not _PROFILED["done"]:
        _PROFILED["done"] = True
        try:
            _prep()
            _seq()
            import torch as _t
            reps = 20
            e0, e1 = _t.cuda.Event(True), _t.cuda.Event(True)
            e0.record()
            for _ in range(reps):
                _prep()
            e1.record(); _t.cuda.synchronize()
            prep_us = e0.elapsed_time(e1) * 1000 / reps
            e0.record()
            for _ in range(reps):
                _seq()
            e1.record(); _t.cuda.synchronize()
            seq_us = e0.elapsed_time(e1) * 1000 / reps
            print(f"KDA_PROFILE: prep={prep_us:.1f}us seq={seq_us:.1f}us NT={NT} B*H={B*H}", flush=True)
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"KDA_PROFILE: failed {exc}", flush=True)

    _prep()
    _seq()
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
        return kda_forward(q, k, v, g, beta, scale=self.scale, chunk_size=self.chunk_size)


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