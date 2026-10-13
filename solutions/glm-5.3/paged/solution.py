"""Custom Triton paged-attention decode kernel (flash-decoding style).

Strategy: split each sequence into chunks handled by separate programs
(split-K over the token dimension). Each program computes an online-softmax
partial attention over its chunk for all G query heads sharing one KV head,
then a small combine kernel merges the partial results.
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


@triton.jit
def _paged_decode_split(
    q_ptr, kv_ptr, bt_ptr, sl_ptr,
    pout_ptr, pl_ptr, pm_ptr,
    sm_scale,
    stride_qb, stride_qh,
    stride_kvb, stride_kvt, stride_kvh,
    stride_btb,
    B, H, Hkv, G,
    S,
    D: tl.constexpr,
    P: tl.constexpr,
    NT: tl.constexpr,
    GP: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    pid_s = tl.program_id(2)

    L = tl.load(sl_ptr + pid_b)
    n_start = pid_s * NT
    step = S * NT

    g = tl.arange(0, GP)
    g_mask = g < G
    dq = tl.arange(0, D)
    qh = pid_hkv * G + g
    q_ptrs = q_ptr + pid_b * stride_qb + qh[:, None] * stride_qh + dq[None, :]
    q = tl.load(q_ptrs, mask=g_mask[:, None], other=0.0)

    m_i = tl.full([GP], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([GP], dtype=tl.float32)
    acc = tl.zeros([GP, D], dtype=tl.float32)

    pos = tl.arange(0, NT)
    p = pos // P
    r = pos % P

    for n in range(n_start, L, step):
        tok = n + pos
        tok_mask = tok < L
        page = tl.load(bt_ptr + pid_b * stride_btb + (n // P + p),
                       mask=tok_mask, other=0)
        kv_off = (page[:, None] * stride_kvb + r[:, None] * stride_kvt
                  + pid_hkv * stride_kvh + dq[None, :])
        k = tl.load(kv_ptr + kv_off, mask=tok_mask[:, None], other=0.0)
        v = tl.load(kv_ptr + kv_off + D, mask=tok_mask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(tok_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        pw = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pw, 1)
        acc = acc * alpha[:, None] + tl.dot(pw.to(tl.bfloat16), v)
        m_i = m_new

    BH = B * H
    pout_ptrs = (pout_ptr + pid_s * BH * D + (pid_b * H + qh)[:, None] * D + dq[None, :])
    tl.store(pout_ptrs, acc, mask=g_mask[:, None])
    pl_ptrs = pl_ptr + pid_s * BH + pid_b * H + qh
    pm_ptrs = pm_ptr + pid_s * BH + pid_b * H + qh
    tl.store(pl_ptrs, l_i, mask=g_mask)
    tl.store(pm_ptrs, m_i, mask=g_mask)


@triton.jit
def _paged_decode_combine(
    pout_ptr, pl_ptr, pm_ptr, out_ptr,
    B, H, S,
    D: tl.constexpr,
    SP: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = pid
    s = tl.arange(0, SP)
    s_mask = s < S
    d = tl.arange(0, D)

    m = tl.load(pm_ptr + s * B * H + bh, mask=s_mask, other=float("-inf"))
    l = tl.load(pl_ptr + s * B * H + bh, mask=s_mask, other=0.0)
    m_star = tl.max(m, 0)
    w = tl.math.exp2(m - m_star)
    w = tl.where(s_mask, w, 0.0)

    acc = tl.load(pout_ptr + s[:, None] * B * H * D + bh * D + d[None, :],
                  mask=s_mask[:, None], other=0.0)
    num = tl.sum(acc * w[:, None], 0)
    den = tl.sum(w * l, 0)
    out = num / den
    tl.store(out_ptr + bh * D + d, out.to(tl.bfloat16))


class Model(nn.Module):
    def __init__(self, batch, num_heads, num_kv_heads, head_dim, seq_len, page_size):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.batch = batch
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.page_size = page_size
        self.group_size = num_heads // num_kv_heads
        self.scale = math.log2(math.e) / math.sqrt(head_dim)  # exp2-domain scale
        self.register_buffer("_dummy", torch.zeros(1, dtype=torch.bfloat16), persistent=False)

        # Split factor: aim for >= a few hundred programs so we saturate the GPU.
        bh = batch * num_kv_heads
        self.num_splits = max(1, min(
            (seq_len + 127) // 128,
            (4096 + bh - 1) // bh,
        ))
        self._ws_key = None
        self._ws = None

    def _workspace(self, S, B, H, D, device):
        key = (S, B, H, D, str(device))
        if self._ws_key != key:
            self._ws = (
                torch.empty(S, B, H, D, dtype=torch.float32, device=device),
                torch.empty(S, B, H, dtype=torch.float32, device=device),
                torch.empty(S, B, H, dtype=torch.float32, device=device),
            )
            self._ws_key = key
        return self._ws

    def forward(self, query, kv_cache, block_table, seq_lens):
        B, H, D = query.shape
        Hkv = self.num_kv_heads
        G = self.group_size
        P = self.page_size
        S = self.num_splits

        pout, pl, pm = self._workspace(S, B, H, D, query.device)

        grid = (B, Hkv, S)
        _paged_decode_split[grid](
            query, kv_cache, block_table, seq_lens,
            pout, pl, pm,
            self.scale,
            query.stride(0), query.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            B, H, Hkv, G,
            S,
            D=D, P=P, NT=128, GP=16,
            num_warps=8, num_stages=2,
        )

        out = torch.empty(B, H, D, dtype=query.dtype, device=query.device)
        SP = triton.next_power_of_2(S)
        _paged_decode_combine[(B * H,)](
            pout, pl, pm, out,
            B, H, S,
            D=D, SP=SP,
            num_warps=1,
        )
        return out


def get_inputs():
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
