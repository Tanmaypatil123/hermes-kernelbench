"""Custom Triton top-k over the last dim of a 2D fp32 tensor.

Approach
--------
* Pack (value, index) into a single int64 key whose signed ordering matches
  descending float ordering (sign-magnitude -> sortable transform on the fp32
  bit pattern), so one selection pass recovers both values and indices.
* Selection is done by iterated packed-max extraction (cheap shuffle-based
  reductions) instead of a full bitonic sort of the segment.
* Fused single-launch kernel (k > 1): grid = (batch * splits). Each program
  extracts the top-KP keys of one SEG-sized slice of a row into a scratch
  buffer, then bumps a per-row arrival counter. The last program to arrive
  for a row merges all segment winners by the same extraction and writes the
  final top-k values / indices, then resets the counter for the next call.
* k == 1 degenerates to a single-pass packed-argmax reduction.
* After the first call, launches go through the cached pre-compiled kernel
  runner to cut host-side dispatch overhead (with fallback to the standard
  dispatch path if the fast path is unavailable).
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

OP_TYPE = "topk"
SUPPORTED_PRECISIONS = ["fp32"]

_MIN64 = -9223372036854775808


@triton.jit
def _pack_keys(v, offs, IBITS: tl.constexpr):
    # fp32 bits -> monotonic signed int32 key, then pack index in low bits.
    vi = v.to(tl.int32, bitcast=True)
    key32 = vi ^ ((vi >> 31) & 0x7FFFFFFF)
    return (key32.to(tl.int64) << IBITS) | offs.to(tl.int64)


@triton.jit
def _unpack_val(m, IBITS: tl.constexpr):
    key32 = (m >> IBITS).to(tl.int32)
    u = key32 ^ ((key32 >> 31) & 0x7FFFFFFF)
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _fused_kernel(x_ptr, cand_ptr, cnt_ptr, v_ptr, i_ptr, n, k, splits,
                  KP: tl.constexpr, BLOCK: tl.constexpr, MSIZE: tl.constexpr,
                  IBITS: tl.constexpr, IMASK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // splits
    seg = pid % splits
    offs = seg * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + row.to(tl.int64) * n + offs, mask=mask,
                other=-float("inf"))
    p = _pack_keys(v, offs, IBITS)
    p = tl.where(mask, p, -9223372036854775808)

    cand_row = cand_ptr + row.to(tl.int64) * (splits * KP)
    base = cand_row + seg * KP
    for j in tl.static_range(KP):
        m = tl.max(p, axis=0)
        tl.store(base + j, m)
        p = tl.where(p == m, -9223372036854775808, p)

    # arrival counter; the last program for this row performs the merge.
    old = tl.atomic_add(cnt_ptr + row, 1, sem="acq_rel", scope="gpu")
    if old == splits - 1:
        tl.store(cnt_ptr + row, 0)  # reset for the next launch
        carr = tl.arange(0, MSIZE)
        c = tl.load(cand_row + carr, mask=carr < splits * KP,
                    other=-9223372036854775808)
        outv = v_ptr + row.to(tl.int64) * k
        outi = i_ptr + row.to(tl.int64) * k
        for j in tl.static_range(KP):
            m2 = tl.max(c, axis=0)
            if j < k:
                tl.store(outv + j, _unpack_val(m2, IBITS))
                tl.store(outi + j, m2 & IMASK)
            c = tl.where(c == m2, -9223372036854775808, c)


@triton.jit
def _argmax1_kernel(x_ptr, v_ptr, i_ptr, n,
                    BLOCK: tl.constexpr, IBITS: tl.constexpr,
                    IMASK: tl.constexpr):
    row = tl.program_id(0)
    base = x_ptr + row.to(tl.int64) * n
    best = tl.full([1], -9223372036854775808, tl.int64)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(base + offs, mask=mask, other=-float("inf"))
        p = _pack_keys(v, offs, IBITS)
        p = tl.where(mask, p, -9223372036854775808)
        cm = tl.max(p, axis=0)
        best = tl.maximum(best, tl.expand_dims(cm, 0))
    vals = _unpack_val(best, IBITS)
    idx = best & IMASK
    one = tl.arange(0, 1)
    tl.store(v_ptr + row + one, vals)
    tl.store(i_ptr + row + one, idx)


class Model(nn.Module):
    """Top-k over the last dim of a 2D tensor.

    Input:
        x: (batch, n) fp32
    Output:
        values:  (batch, k) fp32, sorted descending
        indices: (batch, k) int64, into the last dim of x
    """

    def __init__(self, batch: int, n: int, k: int):
        super().__init__()
        self.batch, self.n, self.k = int(batch), int(n), int(k)
        # Dummy buffer so state_dict is non-empty (mirrors the reference).
        self.register_buffer("_dummy", torch.zeros(1))

        assert self.k <= self.n and self.k >= 1
        ibits = max(20, int(self.n).bit_length() + 1)
        assert ibits <= 31
        self.IBITS = ibits
        self.IMASK = (1 << ibits) - 1

        self.KP = max(2, triton.next_power_of_2(self.k))
        seg = 2048
        while seg < self.KP * 2:
            seg *= 2
        # keep the per-row merge bounded
        while ((self.n + seg - 1) // seg) * self.KP > 8192:
            seg *= 2
        self.SEG = seg
        self.splits = (self.n + seg - 1) // seg
        self.total = self.splits * self.KP
        self.MSIZE = triton.next_power_of_2(self.total)
        self._tk_scratch = {}
        self._runners = {}

    def _tk_buf(self, dev):
        buf = self._tk_scratch.get(dev)
        if buf is None:
            cand = torch.empty(self.batch * self.total, dtype=torch.int64,
                               device=dev)
            cnt = torch.zeros(self.batch, dtype=torch.int32, device=dev)
            buf = (cand, cnt)
            self._tk_scratch[dev] = buf
        return buf

    def _normal_launch(self, fn, grid0, args, consts):
        fn[(grid0,)](*args, **consts, num_warps=8)

    def forward(self, x: torch.Tensor):
        x = x.contiguous()
        batch, n, k = self.batch, self.n, self.k
        dev = x.device
        vals = torch.empty((batch, k), dtype=torch.float32, device=dev)
        idxs = torch.empty((batch, k), dtype=torch.int64, device=dev)
        if k == 1:
            fn = _argmax1_kernel
            args = (x, vals, idxs, n)
            consts = dict(BLOCK=4096, IBITS=self.IBITS, IMASK=self.IMASK)
            tail = (4096, self.IBITS, self.IMASK)
            grid0 = batch
        else:
            cand, cnt = self._tk_buf(dev)
            fn = _fused_kernel
            args = (x, cand, cnt, vals, idxs, n, k, self.splits)
            consts = dict(KP=self.KP, BLOCK=self.SEG, MSIZE=self.MSIZE,
                          IBITS=self.IBITS, IMASK=self.IMASK)
            tail = (self.KP, self.SEG, self.MSIZE, self.IBITS, self.IMASK)
            grid0 = batch * self.splits

        key = (id(fn), dev.index)
        st = self._runners.get(key)
        if st is False:
            self._normal_launch(fn, grid0, args, consts)
        elif st is None:
            self._normal_launch(fn, grid0, args, consts)
            try:
                comp = fn.warmup(*args, grid=(1,), num_warps=8, **consts)
                self._runners[key] = comp[(grid0, 1, 1)]
            except Exception:
                self._runners[key] = False
        else:
            try:
                st(*args, *tail)
            except Exception:
                self._runners[key] = False
                self._normal_launch(fn, grid0, args, consts)
        return vals, idxs


# Module-level shims rebuilt by check.py / benchmark.py per shape.
batch = 64
n = 8192
k = 8


def get_inputs():
    x = torch.randn(batch, n, dtype=torch.float32)
    return [x]


def get_init_inputs():
    return [batch, n, k]
