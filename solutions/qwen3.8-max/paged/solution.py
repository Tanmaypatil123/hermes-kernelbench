"""Paged attention decode — custom Triton split-KV (flash-decoding) kernel.

Single-query decode with GQA and a paged KV cache laid out as
(num_blocks, page_size, num_kv_heads, head_dim * 2), last dim packing [K | V].

Design (single fused kernel):
  grid (B, num_kv_heads, S). Each program handles all G query heads of one KV
  head over a CHUNK-token slice of the sequence (split-KV for occupancy).
  G rows are zero-padded to M=16 so tl.dot uses tensor cores. Online softmax
  in the exp2 domain, fp32 accumulation.
  After writing its partial (acc, m, l) to an fp32 workspace, each program
  takes an atomic ticket; the LAST program of each (b, kv_head) group merges
  all S partials (online LSE merge) and writes the bf16 output, then resets
  the counter so the workspace is reusable (CUDA-graph friendly).

Repeated calls with identical input pointers replay a captured CUDA graph to
eliminate launch overhead; any mismatch falls back to a direct launch.

No SDPA / vLLM / FlashInfer anywhere; everything is custom Triton.
"""
import math

import torch
import triton
import triton.language as tl
import torch.nn as nn

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

LOG2E = 1.4426950408889634


@triton.jit
def _paged_decode_fused(
    q_ptr, kv_ptr, bt_ptr, sl_ptr,
    out_ptr, po_ptr, pm_ptr, pl_ptr, cnt_ptr,
    stride_qb, stride_qh,
    stride_kvb, stride_kvp, stride_kvh,
    stride_btb,
    H, Hkv, G,
    qk_scale,
    S, CHUNK,
    D: tl.constexpr,
    P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPAD: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    s = tl.program_id(2)

    offs_hb = tl.arange(0, 16)
    offs_d = tl.arange(0, D)
    mask_g = offs_hb < G

    # q tile: (16, D), rows beyond G zero-padded so tl.dot gets M=16.
    q = tl.load(
        q_ptr + b * stride_qb + (hk * G + offs_hb)[:, None] * stride_qh + offs_d[None, :],
        mask=mask_g[:, None], other=0.0,
    )

    L = tl.load(sl_ptr + b)
    start = s * CHUNK
    end = tl.minimum(start + CHUNK, L)

    m_i = tl.full([16], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([16], dtype=tl.float32)
    acc = tl.zeros([16, D], dtype=tl.float32)

    bt_base = bt_ptr + b * stride_btb
    for n0 in tl.range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < end
        phys = tl.load(bt_base + offs_n // P, mask=mask_n, other=0).to(tl.int64)
        base = phys * stride_kvb + (offs_n % P) * stride_kvp + hk * stride_kvh
        k = tl.load(kv_ptr + base[:, None] + offs_d[None, :])            # (BLOCK_N, D)
        v = tl.load(kv_ptr + base[:, None] + (D + offs_d)[None, :])      # (BLOCK_N, D)

        qk = tl.dot(q, tl.trans(k)) * qk_scale                           # (16, BLOCK_N)
        qk = tl.where(mask_n[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    head = hk * G + offs_hb
    po_ptrs = po_ptr + (b * H + head)[:, None] * (S * D) + s * D + offs_d[None, :]
    tl.store(po_ptrs, acc, mask=mask_g[:, None])
    tl.store(pm_ptr + (b * H + head) * S + s, m_i, mask=mask_g)
    tl.store(pl_ptr + (b * H + head) * S + s, l_i, mask=mask_g)

    # ---- last-CTA merge over the S splits of this (b, kv_head) group ----
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr + b * Hkv + hk, 1, sem="acq_rel", scope="gpu")
    if ticket == S - 1:
        offs_s = tl.arange(0, SPAD)
        mask_s = offs_s < S
        for g in tl.range(0, G):
            h = hk * G + g
            pbase = (b * H + h) * S
            m = tl.load(pm_ptr + pbase + offs_s, mask=mask_s, other=float("-inf"),
                        cache_modifier=".cg")
            l = tl.load(pl_ptr + pbase + offs_s, mask=mask_s, other=0.0,
                        cache_modifier=".cg")
            M = tl.max(m, 0)
            M = tl.where(M == float("-inf"), 0.0, M)
            w = tl.math.exp2(m - M)
            l_tot = tl.sum(w * l, 0)
            acc_s = tl.load(po_ptr + pbase * D + offs_s[:, None] * D + offs_d[None, :],
                            mask=mask_s[:, None], other=0.0, cache_modifier=".cg")
            o = tl.sum(acc_s * w[:, None], 0) / tl.maximum(l_tot, 1e-30)
            tl.store(out_ptr + (b * H + h) * D + offs_d, o.to(out_ptr.dtype.element_ty))
        # reset the ticket counter so the buffer is clean for the next launch
        tl.store(cnt_ptr + b * Hkv + hk, 0)


class Model(nn.Module):
    """Single-query paged attention decode (fused Triton split-KV flash-decoding).

    Forward inputs (all on device):
      query:       (batch, num_heads, head_dim)               bf16
      kv_cache:    (num_blocks, page_size, num_kv_heads, head_dim * 2)  bf16
      block_table: (batch, max_blocks)                        int32
      seq_lens:    (batch,)                                   int32
    Output:
      attn_out:    (batch, num_heads, head_dim)               bf16
    """

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

        # launch-plan / workspace / graph caches
        self._TARGET_CTAS = 768
        self._MIN_CHUNK = 128
        self._plan_key = None
        self._ws_key = None
        self._graphs = {}
        self._seen = {}
        self._graphs_ok = True

    def _plan(self, B, Hkv, L, D):
        BLOCK_N = 64 if D <= 64 else 32
        base = max(1, B * Hkv)
        s_target = min(
            max(1, (self._TARGET_CTAS + base - 1) // base),
            max(1, (L + self._MIN_CHUNK - 1) // self._MIN_CHUNK),
            16,
        )
        chunk = (L + s_target - 1) // s_target
        chunk = ((chunk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        S = max(1, (L + chunk - 1) // chunk)
        SPAD = 1 << (S - 1).bit_length()
        return S, chunk, BLOCK_N, SPAD

    def _workspace(self, B, H, Hkv, S, D, device):
        key = (B, H, Hkv, S, D, str(device))
        if self._ws_key != key:
            n_acc = B * H * S * D
            n_ml = B * H * S
            buf = torch.empty(n_acc + 2 * n_ml, dtype=torch.float32, device=device)
            cnt = torch.zeros(B * Hkv, dtype=torch.int32, device=device)
            self._ws = (buf[:n_acc], buf[n_acc:n_acc + n_ml], buf[n_acc + n_ml:], cnt)
            self._ws_key = key
        return self._ws

    def _launch(self, query, kv_cache, block_table, seq_lens, out, ws):
        B, H, D = query.shape
        Hkv = self.num_kv_heads
        S, CHUNK, BLOCK_N, SPAD = self._S, self._CHUNK, self._BLOCK_N, self._SPAD
        po, pm, pl, cnt = ws
        _paged_decode_fused[(B, Hkv, S)](
            query, kv_cache, block_table, seq_lens,
            out, po, pm, pl, cnt,
            query.stride(0), query.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            H, Hkv, self.group_size,
            self.scale * LOG2E,
            S, CHUNK,
            D=D, P=self.page_size, BLOCK_N=BLOCK_N, SPAD=SPAD,
            num_warps=4, num_stages=3,
        )

    def forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        B, H, D = query.shape
        L = self.seq_len

        key = (B, H, D, L, self.num_kv_heads, self.page_size)
        if self._plan_key != key:
            self._plan_key = key
            self._S, self._CHUNK, self._BLOCK_N, self._SPAD = self._plan(
                B, self.num_kv_heads, L, D)
        S = self._S

        ws = self._workspace(B, H, self.num_kv_heads, S, D, query.device)

        # CUDA-graph fast path: replay when we see the exact same inputs again.
        gkey = (query.data_ptr(), kv_cache.data_ptr(), block_table.data_ptr(),
                seq_lens.data_ptr(), query.shape, kv_cache.shape, block_table.shape)
        entry = self._graphs.get(gkey)
        if entry is not None:
            entry[0].replay()
            return entry[1]

        seen = self._seen.get(gkey, 0)
        if self._graphs_ok and seen >= 1 and len(self._graphs) < 24:
            try:
                out = torch.empty(B, H, D, dtype=query.dtype, device=query.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._launch(query, kv_cache, block_table, seq_lens, out, ws)
                self._graphs[gkey] = (graph, out)
                graph.replay()
                return out
            except Exception:
                self._graphs_ok = False
        self._seen[gkey] = seen + 1

        out = torch.empty(B, H, D, dtype=query.dtype, device=query.device)
        self._launch(query, kv_cache, block_table, seq_lens, out, ws)
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
