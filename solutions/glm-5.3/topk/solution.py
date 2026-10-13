"""Custom top-k for RTX PRO 6000 (SM120), implemented from scratch in Triton.

One CTA per row. Sampled-pivot selection with atomic-counter compaction:
  1. Strided-sample the row (SUB=1024) and extract the top few distinct
     sample values by iterative max-reduction.
  2. Walk down the candidate pivots; for each, one streaming pass gathers all
     elements >= pivot into a scratch buffer using a global atomic counter for
     slot assignment (no scans, no per-tile reductions).
  3. Bitonic-sort the <= S candidates in registers (value,index) int64 keys,
     emit the top-k.
  4. If the pivot walk fails (duplicate-heavy data), fall back to an exact
     multi-level radix-histogram refinement with a capped tie-aware gather.

The order-preserving map v = bits ^ 0x7FFFFFFF (negatives) / bits (positives)
turns fp32 comparisons into int32 comparisons.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gather_atomic(base, n, bound, scratch_ptr, cnt_ptr, row,
                   S2: tl.constexpr, BLOCK: tl.constexpr):
    # reset slot counter, gather all v >= bound into scratch[row]
    zzz = n - n
    tl.store(cnt_ptr + row, zzz)
    tl.debug_barrier()
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(base + offs, mask=m, other=0.0)
        bits = x.to(tl.int32, bitcast=True)
        v = tl.where(bits < 0, bits ^ 2147483647, bits)
        keep = m & (v >= bound)
        ki = keep.to(tl.int32)
        cntp = cnt_ptr + row + tl.zeros((BLOCK,), tl.int32)
        pos = tl.atomic_add(cntp, ki, mask=keep)
        key = (v.to(tl.int64) << 32) | offs.to(tl.int64)
        tl.store(scratch_ptr + row * S2 + pos, key, mask=keep & (pos < S2))
    tl.debug_barrier()
    return tl.atomic_add(cnt_ptr + row, zzz)


@triton.jit
def _emit(scratch_ptr, outv_ptr, outi_ptr, row, carry, k,
          STRIDE: tl.constexpr, SZ: tl.constexpr):
    so = tl.arange(0, SZ)
    keys = tl.load(scratch_ptr + row * STRIDE + so, mask=so < carry, other=0)
    keys = tl.sort(keys, descending=True)
    v32 = (keys >> 32).to(tl.int32)
    bo = tl.where(v32 < 0, v32 ^ 2147483647, v32)
    vals_o = bo.to(tl.float32, bitcast=True)
    inds = keys & 0xFFFFFFFF
    om = so < k
    tl.store(outv_ptr + row * k + so, vals_o, mask=om)
    tl.store(outi_ptr + row * k + so, inds, mask=om)


@triton.jit
def _topk_kernel(
    x_ptr, outv_ptr, outi_ptr, scratch_ptr, cnt_ptr,
    n, k,
    S1: tl.constexpr, S2: tl.constexpr, SUB: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = x_ptr + row * n
    zero = k - k
    a8 = tl.arange(0, 8)

    # ---------- candidates: top distinct sample values ----------
    sidx = (tl.arange(0, SUB).to(tl.int64) * n) // SUB
    xs = tl.load(base + sidx)
    sb = xs.to(tl.int32, bitcast=True)
    rem = tl.where(sb < 0, sb ^ 2147483647, sb)
    cand = tl.zeros((8,), tl.int32)
    i0 = (2 * k * SUB + n - 1) // n - 1
    i0 = tl.minimum(7, tl.maximum(0, i0))
    E = tl.minimum(8, i0 + 4)
    for i in range(0, E):
        mx = tl.max(rem, 0)
        cand = tl.where(a8 == i, mx, cand)
        rem = tl.where(rem == mx, -2147483648, rem)

    # ---------- pivot walk-down ----------
    carry = zero
    idx = i0
    bound = zero
    while (carry < k) & (idx < E):
        bound = tl.sum(tl.where(a8 == idx, cand, 0))
        carry = _gather_atomic(base, n, bound, scratch_ptr, cnt_ptr, row, S2, BLOCK)
        idx += 1

    cap = zero + S2
    if (carry < k) | (carry > S2):
        # ---------- fallback: multi-level radix histogram ----------
        idxr = tl.arange(0, 1024)
        hist = tl.zeros((1024,), tl.int32)
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < n
            x = tl.load(base + offs, mask=m, other=0.0)
            bits = x.to(tl.int32, bitcast=True)
            v = tl.where(bits < 0, bits ^ 2147483647, bits)
            v = tl.where(m, v, -2147483648)
            hist += tl.histogram((v >> 22) + 512, 1024)
            pad = BLOCK - tl.sum(m.to(tl.int32))
            hist = hist - tl.where(idxr == 0, pad, 0)
        total = tl.sum(hist, 0)
        cum = tl.cumsum(hist, 0)
        suf = total - cum + hist
        t = tl.max(tl.where(suf >= k, idxr, -1))
        c_hi = tl.sum(tl.where(idxr > t, hist, 0))
        bound = (t - 512) * 4194304
        cnt = tl.sum(tl.where(idxr == t, hist, 0))
        done = ((c_hi + cnt) <= S1).to(tl.int32)

        if done == 0:
            need = k - c_hi
            h2 = tl.zeros((1024,), tl.int32)
            for start in range(0, n, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                m = offs < n
                x = tl.load(base + offs, mask=m, other=0.0)
                bits = x.to(tl.int32, bitcast=True)
                v = tl.where(bits < 0, bits ^ 2147483647, bits)
                sel2 = m & (v >= bound) & (v < bound + 4194304)
                b2 = tl.where(sel2, (v >> 12) & 1023, 0)
                h2 += tl.histogram(b2, 1024)
                pad = BLOCK - tl.sum(sel2.to(tl.int32))
                h2 = h2 - tl.where(idxr == 0, pad, 0)
            tot2 = tl.sum(h2, 0)
            cum2 = tl.cumsum(h2, 0)
            suf2 = tot2 - cum2 + h2
            t2 = tl.max(tl.where(suf2 >= need, idxr, -1))
            c_hi += tl.sum(tl.where(idxr > t2, h2, 0))
            bound = bound | (t2 << 12)
            cnt = tl.sum(tl.where(idxr == t2, h2, 0))
            done = ((c_hi + cnt) <= S1).to(tl.int32)

        if done == 0:
            need = k - c_hi
            h3 = tl.zeros((1024,), tl.int32)
            for start in range(0, n, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                m = offs < n
                x = tl.load(base + offs, mask=m, other=0.0)
                bits = x.to(tl.int32, bitcast=True)
                v = tl.where(bits < 0, bits ^ 2147483647, bits)
                sel3 = m & (v >= bound) & (v < bound + 4096)
                b3 = tl.where(sel3, (v >> 2) & 1023, 0)
                h3 += tl.histogram(b3, 1024)
                pad = BLOCK - tl.sum(sel3.to(tl.int32))
                h3 = h3 - tl.where(idxr == 0, pad, 0)
            tot3 = tl.sum(h3, 0)
            cum3 = tl.cumsum(h3, 0)
            suf3 = tot3 - cum3 + h3
            t3 = tl.max(tl.where(suf3 >= need, idxr, -1))
            c_hi += tl.sum(tl.where(idxr > t3, h3, 0))
            bound = bound | (t3 << 2)
            cnt = tl.sum(tl.where(idxr == t3, h3, 0))
            done = ((c_hi + cnt) <= S1).to(tl.int32)

        if done == 0:
            c1 = zero
            c2 = zero
            c3 = zero
            for start in range(0, n, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                m = offs < n
                x = tl.load(base + offs, mask=m, other=0.0)
                bits = x.to(tl.int32, bitcast=True)
                v = tl.where(bits < 0, bits ^ 2147483647, bits)
                sel4 = m & (v >= bound) & (v < bound + 4)
                c1 += tl.sum((sel4 & (v == bound + 1)).to(tl.int32), 0)
                c2 += tl.sum((sel4 & (v == bound + 2)).to(tl.int32), 0)
                c3 += tl.sum((sel4 & (v == bound + 3)).to(tl.int32), 0)
            thr = bound
            cabove = c_hi + c3 + c2 + c1
            if c_hi + c3 >= k:
                thr = bound + 3
                cabove = c_hi
            if (c_hi + c3 < k) & (c_hi + c3 + c2 >= k):
                thr = bound + 2
                cabove = c_hi + c3
            if (c_hi + c3 + c2 < k) & (c_hi + c3 + c2 + c1 >= k):
                thr = bound + 1
                cabove = c_hi + c3 + c2
            bound = thr
            cap = k - cabove

        # tie-aware gather with rank cap
        carry = zero
        carry_eq = zero
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < n
            x = tl.load(base + offs, mask=m, other=0.0)
            bits = x.to(tl.int32, bitcast=True)
            v = tl.where(bits < 0, bits ^ 2147483647, bits)
            eq = m & (v == bound)
            eqr = tl.cumsum(eq.to(tl.int32), 0) + carry_eq
            keep = (m & (v > bound)) | (eq & (eqr <= cap))
            ki = keep.to(tl.int32)
            pos = tl.cumsum(ki, 0) + carry - 1
            key = (v.to(tl.int64) << 32) | offs.to(tl.int64)
            tl.store(scratch_ptr + row * S2 + pos, key, mask=keep)
            carry += tl.sum(ki, 0)
            carry_eq += tl.sum(eq.to(tl.int32), 0)

    # ---------- sort & emit ----------
    if carry > S1:
        _emit(scratch_ptr, outv_ptr, outi_ptr, row, carry, k, S2, S2)
    else:
        _emit(scratch_ptr, outv_ptr, outi_ptr, row, carry, k, S2, S1)


def _topk(x, k):
    batch, n = x.shape
    S1 = max(64, 4 * triton.next_power_of_2(k))
    S2 = 4 * S1
    scratch = torch.empty((batch, S2), dtype=torch.int64, device=x.device)
    cnt = torch.empty((batch,), dtype=torch.int32, device=x.device)
    values = torch.empty((batch, k), dtype=torch.float32, device=x.device)
    indices = torch.empty((batch, k), dtype=torch.int64, device=x.device)
    _topk_kernel[(batch,)](
        x, values, indices, scratch, cnt, n, k,
        S1=S1, S2=S2, SUB=1024, BLOCK=4096, num_warps=8,
    )
    return values, indices


OP_TYPE = "topk"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]


class Model(nn.Module):
    """Top-k over the last dim of a 2D tensor."""

    def __init__(self, batch: int, n: int, k: int):
        super().__init__()
        self.batch, self.n, self.k = batch, n, k
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(self, x: torch.Tensor):
        x = x.contiguous()
        return _topk(x, self.k)


batch = 64
n = 8192
k = 8


def get_inputs():
    x = torch.randn(batch, n, dtype=torch.float32)
    return [x]


def get_init_inputs():
    return [batch, n, k]
