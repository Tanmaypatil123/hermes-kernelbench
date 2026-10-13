"""Paged-attention decode (single query per batch element) — custom Triton kernel.

Design
------
Decode is bandwidth-bound: the KV cache must be streamed once.  The scheme is a
two-stage split-K flash decode:

  stage 1  grid = (batch * num_kv_heads, num_partitions)
           One program owns a single kv head of a single batch element and *all*
           query heads that share it (the GQA group), so every K/V byte is read
           from DRAM exactly once instead of once per query head.  The group is
           processed as one (GP, D) tile with a single tensor-core ``tl.dot``
           (padded to 16 rows, the MMA minimum M).  Pages are gathered inside
           the kernel from ``block_table``; the tail is predicated.  Each
           program emits per-partition (m, l, acc) in fp32.

  stage 2  grid = (batch * num_kv_heads)
           Merges partitions with a log-sum-exp combine and writes bf16.

Numerics: QK^T is a bf16 x bf16 tensor-core dot accumulated in fp32; the P@V
product casts the fp32 weights and the bf16 values to fp16 (more mantissa than
bf16 for weights confined to [0, 1]) with fp32 accumulation.

Launch overhead is material at these sizes (the smallest shape is ~9 us of pure
DRAM traffic), so identical consecutive calls are captured into a CUDA graph
keyed by the input buffer addresses and replayed.  A graph reads its inputs from
memory at replay time, so this is purely a launch-amortisation; results are
recomputed from whatever the buffers currently hold.
"""

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

OP_TYPE = "attention"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]

# --- Shape knobs (overridden by check.py / benchmark.py from shapes.py) ----
BATCH = 8
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
SEQ_LEN = 1024
PAGE_SIZE = 16

# --- Tuning knobs ----------------------------------------------------------
_BLOCK_L = 64      # tokens per inner loop step (multiple of page_size)
_GP = 16           # padded GQA group size (MMA minimum M)
_TARGET_PROGRAMS = 1024
_PARTLEN_MIN = 160  # keep at least ~3 inner iterations per program
_NEG = -1.0e30


@triton.jit
def _paged_attn_part1(
    q_ptr, kv_ptr, bt_ptr, sl_ptr,
    pm_ptr, pl_ptr, pa_ptr,
    stride_qb, stride_qh, stride_bt,
    H: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    G: tl.constexpr, GP: tl.constexpr,
    PARTLEN: tl.constexpr, BLOCK_L: tl.constexpr, NP: tl.constexpr,
    SCALE: tl.constexpr,
    NEG: tl.constexpr,
):
    pid = tl.program_id(0)
    part = tl.program_id(1)
    b = pid // Hkv
    hkv = pid % Hkv

    L = tl.load(sl_ptr + b)
    start = part * PARTLEN

    offs_g = tl.arange(0, GP)
    offs_d = tl.arange(0, D)
    offs_l = tl.arange(0, BLOCK_L)
    mask_g = offs_g < G

    # Query rows for this kv head's group: q[b, hkv*G + g, :]
    q_ptrs = (q_ptr + b * stride_qb
              + (hkv * G + offs_g)[:, None] * stride_qh
              + offs_d[None, :])
    q = tl.load(q_ptrs, mask=mask_g[:, None], other=0.0)

    m_i = tl.full((GP,), NEG, tl.float32)
    l_i = tl.zeros((GP,), tl.float32)
    acc = tl.zeros((GP, D), tl.float32)

    # kv_cache: (num_blocks, P, Hkv, 2*D) -> token stride inside a page is Hkv*2D,
    # kv-head stride is 2D, page stride is P*Hkv*2D.
    kv_head = 2 * D
    kv_row = Hkv * 2 * D
    kv_page = P * Hkv * 2 * D
    kv_base = kv_ptr + hkv * kv_head

    for i in range(0, PARTLEN, BLOCK_L):
        t = start + i + offs_l
        m_t = t < L
        pg = t // P
        po = t % P
        page = tl.load(bt_ptr + b * stride_bt + pg, mask=m_t, other=0)

        row = kv_base + page * kv_page + po * kv_row                # (BLOCK_L,)
        # Load K already transposed as (D, BLOCK_L): the head dim is contiguous
        # in memory, so this feeds tl.dot's B operand without a layout convert.
        k_t = tl.load(row[None, :] + offs_d[:, None], mask=m_t[None, :], other=0.0)
        # V tile: (BLOCK_L, D), contiguous along the head dim.
        v = tl.load(row[:, None] + D + offs_d[None, :], mask=m_t[:, None], other=0.0)

        s = tl.dot(q, k_t, out_dtype=tl.float32) * SCALE            # (GP, BLOCK_L)
        s = tl.where(m_t[None, :], s, NEG)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(m_t[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
        m_i = m_new

    # Per-partition partials, packed at stride G (only the G real query heads of
    # the group are meaningful) -> a fraction of the split-K reduction traffic.
    base = pid * NP + part
    tl.store(pm_ptr + base * G + offs_g, m_i, mask=mask_g)
    tl.store(pl_ptr + base * G + offs_g, l_i, mask=mask_g)
    tl.store(pa_ptr + base * G * D + offs_g[:, None] * D + offs_d[None, :],
             acc, mask=mask_g[:, None])


@triton.jit
def _paged_attn_part2(
    pm_ptr, pl_ptr, pa_ptr, out_ptr,
    stride_ob, stride_oh,
    Hkv: tl.constexpr, D: tl.constexpr,
    G: tl.constexpr, GP: tl.constexpr, NP: tl.constexpr,
    NEG: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // Hkv
    hkv = pid % Hkv

    offs_g = tl.arange(0, GP)
    offs_d = tl.arange(0, D)
    mask_g = offs_g < G

    m_i = tl.full((GP,), NEG, tl.float32)
    l_i = tl.zeros((GP,), tl.float32)
    acc = tl.zeros((GP, D), tl.float32)

    for p in range(0, NP):
        base = pid * NP + p
        m_p = tl.load(pm_ptr + base * G + offs_g, mask=mask_g, other=0.0)
        l_p = tl.load(pl_ptr + base * G + offs_g, mask=mask_g, other=0.0)
        a_p = tl.load(pa_ptr + base * G * D + offs_g[:, None] * D + offs_d[None, :],
                      mask=mask_g[:, None], other=0.0)

        m_new = tl.maximum(m_i, m_p)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_p - m_new)
        l_i = l_i * alpha + l_p * beta
        acc = acc * alpha[:, None] + a_p * beta[:, None]
        m_i = m_new

    o = acc / l_i[:, None]
    o_ptrs = (out_ptr + b * stride_ob
              + (hkv * G + offs_g)[:, None] * stride_oh
              + offs_d[None, :])
    tl.store(o_ptrs, o.to(out_ptr.dtype.element_ty), mask=mask_g[:, None])


_SCRATCH = {}
_GRAPHS = {}
_GRAPH_MAX = 32


def _scratch(key, shape, dtype, device):
    buf = _SCRATCH.get(key)
    if buf is None or buf.shape != shape or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _SCRATCH[key] = buf
    return buf


class Model(nn.Module):
    """Single-query paged attention decode (custom Triton kernels)."""

    def __init__(
        self,
        batch: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        page_size: int,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads (GQA)"
        self.batch = batch
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.page_size = page_size
        self.group_size = num_heads // num_kv_heads
        self.scale = 1.0 / math.sqrt(head_dim)
        self.register_buffer("_dummy", torch.zeros(1, dtype=torch.bfloat16), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        B, H, D = query.shape
        Hkv = self.num_kv_heads
        G = self.group_size
        P = self.page_size

        # ---- fast path: same contiguous buffers as a previous call ----
        key = (query.data_ptr(), kv_cache.data_ptr(),
               block_table.data_ptr(), seq_lens.data_ptr(),
               B, H, D, Hkv, P,
               kv_cache.shape[0], block_table.shape[1], seq_lens.shape[0])
        hit = _GRAPHS.get(key)
        if hit is not None:
            hit[0].replay()
            return hit[1]

        q = query.contiguous()
        kvc = kv_cache.contiguous()
        bt = block_table.contiguous()
        sl = seq_lens.contiguous()
        device = query.device

        # Upper bound on seq_len taken from the block table itself: avoids a
        # host<->device sync (seq_lens.max().item()) on the critical path.
        lmax = bt.shape[1] * P
        if lmax <= 0:
            return torch.empty(B, H, D, dtype=query.dtype, device=device)

        nprog = B * Hkv
        np_block = max(1, _TARGET_PROGRAMS // max(1, nprog))
        np_len = max(1, lmax // _PARTLEN_MIN)
        np0 = max(1, min(np_block, np_len))
        partlen = -(-lmax // np0)
        partlen = max(_BLOCK_L, partlen)
        partlen = ((partlen + _BLOCK_L - 1) // _BLOCK_L) * _BLOCK_L
        np_ = max(1, -(-lmax // partlen))
        gp = _GP if G <= _GP else triton.next_power_of_2(G)

        skey = ("pa", nprog, np_, G, D, device)
        pm = _scratch(skey + ("m",), (nprog * np_ * G,), torch.float32, device)
        pl = _scratch(skey + ("l",), (nprog * np_ * G,), torch.float32, device)
        pa = _scratch(skey + ("a",), (nprog * np_ * G * D,), torch.float32, device)

        out = torch.empty(B, H, D, dtype=query.dtype, device=device)

        def launch():
            _paged_attn_part1[(nprog, np_)](
                q, kvc, bt, sl,
                pm, pl, pa,
                q.stride(0), q.stride(1), bt.stride(0),
                H=H, Hkv=Hkv, D=D, P=P,
                G=G, GP=gp,
                PARTLEN=partlen, BLOCK_L=_BLOCK_L, NP=np_,
                SCALE=self.scale, NEG=_NEG,
                num_warps=8, num_stages=3,
            )
            _paged_attn_part2[(nprog,)](
                pm, pl, pa, out,
                out.stride(0), out.stride(1),
                Hkv=Hkv, D=D,
                G=G, GP=gp, NP=np_,
                NEG=_NEG,
                num_warps=4,
            )

        launch()  # also compiles the kernels

        # Identical back-to-back calls (the benchmark loop) get graph-replayed.
        # Only cacheable when no input had to be copied, so the captured
        # pointers are the caller's own buffers.
        aligned = (q.data_ptr() == query.data_ptr()
                   and kvc.data_ptr() == kv_cache.data_ptr()
                   and bt.data_ptr() == block_table.data_ptr()
                   and sl.data_ptr() == seq_lens.data_ptr())
        if aligned and not torch.cuda.is_current_stream_capturing():
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                graph.replay()
                if len(_GRAPHS) >= _GRAPH_MAX:
                    _GRAPHS.pop(next(iter(_GRAPHS)))
                _GRAPHS[key] = (graph, out, pm, pl, pa)
            except Exception:
                pass
        return out


def get_inputs():
    """Build random paged inputs for the current module-level shape knobs."""
    B = BATCH
    H = NUM_HEADS
    Hkv = NUM_KV_HEADS
    D = HEAD_DIM
    L = SEQ_LEN
    P = PAGE_SIZE

    pages_per_seq = (L + P - 1) // P
    total_pages = max(B * pages_per_seq + 8, 64)

    query = torch.randn(B, H, D, dtype=torch.bfloat16) * 0.1
    kv_cache = torch.randn(total_pages, P, Hkv, 2 * D, dtype=torch.bfloat16) * 0.1

    perm = torch.randperm(total_pages)[: B * pages_per_seq].reshape(B, pages_per_seq).int()
    block_table = perm.contiguous()
    seq_lens = torch.full((B,), L, dtype=torch.int32)

    return [query, kv_cache, block_table, seq_lens]


def get_init_inputs():
    return [BATCH, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, SEQ_LEN, PAGE_SIZE]