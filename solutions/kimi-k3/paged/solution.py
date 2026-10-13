"""Custom Triton paged-attention decode kernel (split-KV / FlashDecoding style).

No forbidden ops: softmax/QKV reduction is implemented inside Triton kernels.
Layout:
  query:      (B, H, D) bf16
  kv_cache:   (num_blocks, P, Hkv, 2*D) bf16, last dim packs [K | V]
  block_table:(B, max_blocks) int32
  seq_lens:   (B,) int32
"""
import math

import torch
import torch.nn as nn
import triton
import triton.language as tl

OP_TYPE = "attention"
SUPPORTED_PRECISIONS = ["bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000"]

BATCH = 8
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
SEQ_LEN = 1024
PAGE_SIZE = 16


@triton.jit
def _paged_attn_stage1(
    q_ptr, kv_ptr, bt_ptr, sl_ptr,
    pm_ptr, pl_ptr, po_ptr,          # partial m, l, o (fp32)
    stride_qb, stride_qh,
    stride_kvb, stride_kvp, stride_kvh,  # kv: block, page-slot, kv-head
    stride_btb,
    stride_pmb, stride_plb, stride_pob, stride_pos,
    H, G, Lmax, S, P, scale,
    CHUNK: tl.constexpr, TILE_N: tl.constexpr, D: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    b = pid // H
    h = pid % H
    hkv = h // G

    seq_len = tl.load(sl_ptr + b)
    chunk_start = split * CHUNK
    chunk_end = tl.minimum(chunk_start + CHUNK, seq_len)
    if chunk_start >= seq_len:
        # nothing to do; write neutral partial
        sbase = b * H * S + h * S + split
        tl.store(pm_ptr + sbase, float("-inf"))
        tl.store(pl_ptr + sbase, 0.0)
        offs_d = tl.arange(0, D)
        tl.store(po_ptr + sbase * D + offs_d, 0.0)
        return

    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + offs_d).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    kv_base = kv_ptr + hkv * stride_kvh

    for t0 in range(chunk_start, chunk_end, TILE_N):
        offs_t = t0 + tl.arange(0, TILE_N)
        mask_t = offs_t < chunk_end
        page_off = offs_t // P
        slot = offs_t % P
        page = tl.load(bt_ptr + b * stride_btb + page_off, mask=mask_t, other=0)
        tok_addr = kv_base + page * stride_kvb + slot * stride_kvp
        # K tile: (TILE_N, D)
        k = tl.load(tok_addr[:, None] + offs_d[None, :], mask=mask_t[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * scale
        s = tl.where(mask_t, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        # V tile
        v = tl.load(tok_addr[:, None] + D + offs_d[None, :], mask=mask_t[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    sbase = b * H * S + h * S + split
    tl.store(pm_ptr + sbase, m_i)
    tl.store(pl_ptr + sbase, l_i)
    tl.store(po_ptr + sbase * D + offs_d, acc)


@triton.jit
def _paged_attn_combine(
    pm_ptr, pl_ptr, po_ptr, o_ptr,
    stride_ob, stride_oh,
    stride_pos,
    H, S,
    D: tl.constexpr, S_POW2: tl.constexpr,
):
    pid = tl.program_id(0)  # b * H + h
    offs_s = tl.arange(0, S_POW2)
    mask_s = offs_s < S
    base = pid * S

    m = tl.load(pm_ptr + base + offs_s, mask=mask_s, other=float("-inf"))
    l = tl.load(pl_ptr + base + offs_s, mask=mask_s, other=0.0)
    m_max = tl.max(m, axis=0)
    exp_ = tl.where(mask_s, tl.exp(m - m_max), 0.0)
    l_total = tl.sum(exp_ * l, axis=0)

    offs_d = tl.arange(0, D)
    acc = tl.zeros([D], dtype=tl.float32)
    for s in range(S):
        m_i = tl.load(pm_ptr + base + s)
        l_i = tl.load(pl_ptr + base + s)
        coef = tl.exp(m_i - m_max)
        acc_part = tl.load(po_ptr + (base + s) * D + offs_d)
        acc += coef * acc_part
    b = pid // H
    h = pid % H
    out = acc / l_total
    tl.store(o_ptr + b * stride_ob + h * stride_oh + offs_d, out.to(o_ptr.dtype.element_ty))


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
        self.scale = 1.0 / math.sqrt(head_dim)
        self.register_buffer("_dummy", torch.zeros(1, dtype=torch.bfloat16), persistent=False)

    def forward(self, query, kv_cache, block_table, seq_lens):
        B, H, D = query.shape
        P = self.page_size
        L = self.seq_len
        CHUNK = 256
        S = (L + CHUNK - 1) // CHUNK
        S_POW2 = triton.next_power_of_2(S)
        dev = query.device

        pm = torch.empty(B * H * S, dtype=torch.float32, device=dev)
        pl = torch.empty(B * H * S, dtype=torch.float32, device=dev)
        po = torch.empty(B * H * S * D, dtype=torch.float32, device=dev)
        out = torch.empty(B, H, D, dtype=torch.bfloat16, device=dev)

        grid1 = (B * H, S)
        _paged_attn_stage1[grid1](
            query, kv_cache, block_table, seq_lens,
            pm, pl, po,
            query.stride(0), query.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            0, 0, 0, 0,
            H, self.group_size, L, S, P, self.scale,
            CHUNK=CHUNK, TILE_N=64, D=D,
        )
        grid2 = (B * H,)
        _paged_attn_combine[grid2](
            pm, pl, po, out,
            out.stride(0), out.stride(1),
            0,
            H, S,
            D=D, S_POW2=S_POW2,
        )
        return out


def get_inputs():
    B, H, Hkv, D, L, P = BATCH, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, SEQ_LEN, PAGE_SIZE
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
