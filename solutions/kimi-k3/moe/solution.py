"""Grouped-GEMM MoE up-projection with fused SwiGLU (Triton).

Per expert e: h_e = silu(x_e @ W_gate[e]) * (x_e @ W_up[e]).
One Triton program per (m-tile, n-tile) of one expert; both GEMMs share the
A-tile loads and SwiGLU is applied in the epilogue. The tile schedule is
precomputed on-GPU with tensor ops (no host sync in forward).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

_BM = 128  # fixed m-tile so the schedule can be precomputed without host sync


def _cfgs():
    return [
        triton.Config({"BN": 128, "BK": 64, "GM": 8}, num_warps=8, num_stages=4),
        triton.Config({"BN": 128, "BK": 128, "GM": 8}, num_warps=8, num_stages=3),
        triton.Config({"BN": 128, "BK": 128, "GM": 8}, num_warps=8, num_stages=4),
        triton.Config({"BN": 256, "BK": 64, "GM": 8}, num_warps=8, num_stages=2),
        triton.Config({"BN": 256, "BK": 64, "GM": 8}, num_warps=8, num_stages=3),
        triton.Config({"BN": 256, "BK": 128, "GM": 8}, num_warps=8, num_stages=2),
        triton.Config({"BN": 128, "BK": 64, "GM": 4}, num_warps=8, num_stages=4),
        triton.Config({"BN": 128, "BK": 64, "GM": 16}, num_warps=8, num_stages=4),
        triton.Config({"BN": 64, "BK": 128, "GM": 8}, num_warps=4, num_stages=4),
        triton.Config({"BN": 64, "BK": 64, "GM": 8}, num_warps=4, num_stages=4),
        triton.Config({"BN": 512, "BK": 64, "GM": 8}, num_warps=8, num_stages=2),
        triton.Config({"BN": 256, "BK": 128, "GM": 4}, num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=_cfgs(), key=["H", "I"])
@triton.jit
def _moe_up_kernel(
    x_ptr, wg_ptr, wu_ptr, out_ptr,
    tile_expert_ptr, tile_row_ptr, tile_n_ptr, total_ptr,
    H, I, MT_UB,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
):
    pid = tl.program_id(0)

    # Grouped (L2-swizzled) ordering over the (m-tile, n-tile) space.
    nt_n = tl.cdiv(I, BN)
    in_group = GM * nt_n
    group_id = pid // in_group
    first_m = group_id * GM
    group_m = tl.minimum(MT_UB - first_m, GM)
    mt = first_m + (pid % in_group) % group_m
    n_i = (pid % in_group) // group_m

    total = tl.load(total_ptr)
    if mt >= total:
        return

    e_sel = tl.load(tile_expert_ptr + mt)
    row_base = tl.load(tile_row_ptr + mt)
    n_sel = tl.load(tile_n_ptr + mt)

    rm = tl.arange(0, BM)
    rn = n_i * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_mask = rm < n_sel
    n_mask = rn < I

    rows = row_base + rm  # not BM-aligned for ragged offsets: no alignment hint
    rn = tl.max_contiguous(tl.multiple_of(rn, BN), BN)
    a_ptrs = x_ptr + rows[:, None] * H + rk[None, :]
    w_off = e_sel.to(tl.int64) * H * I
    bg_ptrs = wg_ptr + w_off + rk[:, None] * I + rn[None, :]
    bu_ptrs = wu_ptr + w_off + rk[:, None] * I + rn[None, :]

    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, H, BK):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        bg = tl.load(bg_ptrs, mask=n_mask[None, :], other=0.0)
        bu = tl.load(bu_ptrs, mask=n_mask[None, :], other=0.0)
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)
        a_ptrs += BK
        bg_ptrs += BK * I
        bu_ptrs += BK * I

    y = acc_g * tl.sigmoid(acc_g) * acc_u
    o_ptrs = out_ptr + rows[:, None] * I + rn[None, :]
    tl.store(o_ptrs, y.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


@triton.autotune(configs=_cfgs(), key=["H", "I"])
@triton.jit
def _moe_up_kernel_tma(
    x_ptr, wg_ptr, wu_ptr, out_ptr,
    tile_expert_ptr, tile_row_ptr, tile_n_ptr, total_ptr,
    EH, H, I, MT_UB, T_perm,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
):
    pid = tl.program_id(0)
    nt_n = tl.cdiv(I, BN)
    in_group = GM * nt_n
    group_id = pid // in_group
    first_m = group_id * GM
    group_m = tl.minimum(MT_UB - first_m, GM)
    mt = first_m + (pid % in_group) % group_m
    n_i = (pid % in_group) // group_m

    total = tl.load(total_ptr)
    if mt >= total:
        return

    e_sel = tl.load(tile_expert_ptr + mt).to(tl.int32)
    row_base = tl.load(tile_row_ptr + mt)
    n_sel = tl.load(tile_n_ptr + mt)

    desc_x = tl.make_tensor_descriptor(
        x_ptr, shape=[T_perm, H], strides=[H, 1], block_shape=[BM, BK],
    )
    desc_wg = tl.make_tensor_descriptor(
        wg_ptr, shape=[EH, I], strides=[I, 1], block_shape=[BK, BN],
    )
    desc_wu = tl.make_tensor_descriptor(
        wu_ptr, shape=[EH, I], strides=[I, 1], block_shape=[BK, BN],
    )
    desc_o = tl.make_tensor_descriptor(
        out_ptr, shape=[T_perm, I], strides=[I, 1], block_shape=[BM, BN],
    )

    w_row = e_sel * H
    n0 = n_i * BN
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, H, BK):
        a = desc_x.load([row_base, k])
        bg = desc_wg.load([w_row + k, n0])
        bu = desc_wu.load([w_row + k, n0])
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)

    y = acc_g * tl.sigmoid(acc_g) * acc_u
    # TMA store clamps OOB rows/cols; mask the ragged tail explicitly.
    if n_sel < BM:
        rm = tl.arange(0, BM)
        rn = n0 + tl.arange(0, BN)
        n_mask = rn < I
        rows = row_base + rm
        tl.store(out_ptr + rows[:, None] * I + rn[None, :], y.to(tl.bfloat16),
                 mask=(rm < n_sel)[:, None] & n_mask[None, :])
    else:
        desc_o.store([row_base, n0], y.to(tl.bfloat16))


@triton.autotune(configs=_cfgs(), key=["H", "I"])
@triton.jit
def _moe_up_kernel_ws(
    x_ptr, wg_ptr, wu_ptr, out_ptr,
    tile_expert_ptr, tile_row_ptr, tile_n_ptr, total_ptr,
    EH, H, I, MT_UB, T_perm,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
):
    pid = tl.program_id(0)
    nt_n = tl.cdiv(I, BN)
    in_group = GM * nt_n
    group_id = pid // in_group
    first_m = group_id * GM
    group_m = tl.minimum(MT_UB - first_m, GM)
    mt = first_m + (pid % in_group) % group_m
    n_i = (pid % in_group) // group_m

    total = tl.load(total_ptr)
    if mt >= total:
        return

    e_sel = tl.load(tile_expert_ptr + mt).to(tl.int32)
    row_base = tl.load(tile_row_ptr + mt)
    n_sel = tl.load(tile_n_ptr + mt)

    desc_x = tl.make_tensor_descriptor(
        x_ptr, shape=[T_perm, H], strides=[H, 1], block_shape=[BM, BK],
    )
    desc_wg = tl.make_tensor_descriptor(
        wg_ptr, shape=[EH, I], strides=[I, 1], block_shape=[BK, BN],
    )
    desc_wu = tl.make_tensor_descriptor(
        wu_ptr, shape=[EH, I], strides=[I, 1], block_shape=[BK, BN],
    )
    desc_o = tl.make_tensor_descriptor(
        out_ptr, shape=[T_perm, I], strides=[I, 1], block_shape=[BM, BN],
    )

    w_row = e_sel * H
    n0 = n_i * BN
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, H, BK, warp_specialize=True):
        a = desc_x.load([row_base, k])
        bg = desc_wg.load([w_row + k, n0])
        bu = desc_wu.load([w_row + k, n0])
        acc_g = tl.dot(a, bg, acc_g)
        acc_u = tl.dot(a, bu, acc_u)

    y = acc_g * tl.sigmoid(acc_g) * acc_u
    if n_sel < BM:
        rm = tl.arange(0, BM)
        rn = n0 + tl.arange(0, BN)
        n_mask = rn < I
        rows = row_base + rm
        tl.store(out_ptr + rows[:, None] * I + rn[None, :], y.to(tl.bfloat16),
                 mask=(rm < n_sel)[:, None] & n_mask[None, :])
    else:
        desc_o.store([row_base, n0], y.to(tl.bfloat16))


# 0=try WS-TMA (disabled: crashes Triton's MLIR pipeline on sm120), 1=try TMA, 2=classic
_TMA_STATE = {"level": 1, "alloc_set": False}


def _ws_launch(model, x, out, tile_expert, tile_row, tile_n, total, H, T_perm, mt_ub):
    if not _TMA_STATE["alloc_set"]:
        def _alloc(size: int, alignment: int, stream):
            return torch.empty(size, dtype=torch.int8, device=x.device)
        triton.set_allocator(_alloc)
        _TMA_STATE["alloc_set"] = True
    grid = lambda META: (mt_ub * triton.cdiv(model.I, META["BN"]),)  # noqa: E731
    _moe_up_kernel_ws[grid](
        x, model.W_gate, model.W_up, out,
        tile_expert, tile_row, tile_n, total,
        model.E * H, H, model.I, mt_ub, T_perm,
        BM=_BM,
    )


def _tma_launch(model, x, out, tile_expert, tile_row, tile_n, total, H, T_perm, mt_ub):
    if not _TMA_STATE["alloc_set"]:
        def _alloc(size: int, alignment: int, stream):
            return torch.empty(size, dtype=torch.int8, device=x.device)
        triton.set_allocator(_alloc)
        _TMA_STATE["alloc_set"] = True
    grid = lambda META: (mt_ub * triton.cdiv(model.I, META["BN"]),)  # noqa: E731
    _moe_up_kernel_tma[grid](
        x, model.W_gate, model.W_up, out,
        tile_expert, tile_row, tile_n, total,
        model.E * H, H, model.I, mt_ub, T_perm,
        BM=_BM,
    )


class Model(nn.Module):
    def __init__(self, T_total: int, H: int, I: int, E: int, K: int):  # noqa: E741
        super().__init__()
        self.T_total = T_total
        self.H = H
        self.I = I
        self.E = E
        self.K = K
        self.W_gate = nn.Parameter(torch.empty(E, H, I, dtype=torch.bfloat16))
        self.W_up = nn.Parameter(torch.empty(E, H, I, dtype=torch.bfloat16))
        nn.init.normal_(self.W_gate, std=0.02)
        nn.init.normal_(self.W_up, std=0.02)

    def forward(self, hidden_states: torch.Tensor, expert_offsets: torch.Tensor) -> torch.Tensor:
        T_perm, H = hidden_states.shape
        dev = hidden_states.device
        out = torch.empty(T_perm, self.I, dtype=torch.bfloat16, device=dev)
        x = hidden_states.contiguous()
        offs = expert_offsets.contiguous()

        # On-GPU tile schedule (all small tensors, no host sync).
        counts = (offs[1:] - offs[:-1]).to(torch.int32)                      # (E,)
        mtiles = (counts + (_BM - 1)) // _BM                                 # (E,)
        cum = torch.cumsum(mtiles, 0, dtype=torch.int32)                     # (E,)
        mt_ub = T_perm // _BM + 1 + self.E                                   # upper bound on m-tiles
        idx = torch.arange(mt_ub, device=dev, dtype=torch.int32)
        tile_expert = torch.searchsorted(cum, idx, right=True).clamp_(max=self.E - 1)
        prev = torch.zeros_like(cum)
        prev[1:] = cum[:-1]
        tile_local = idx - prev[tile_expert]
        row_starts = offs[:-1].to(torch.int32)                               # (E,)
        tile_row = row_starts[tile_expert] + tile_local * _BM                # (mt_ub,)
        tile_n = (offs[1:].to(torch.int32)[tile_expert] - tile_row).clamp_(min=0)  # rows valid in tile
        total = cum[-1:].contiguous()                                        # (1,)

        grid = lambda META: (mt_ub * triton.cdiv(self.I, META["BN"]),)       # noqa: E731
        while _TMA_STATE["level"] < 2:
            try:
                if _TMA_STATE["level"] == 0:
                    _ws_launch(self, x, out, tile_expert, tile_row, tile_n, total,
                               H, T_perm, mt_ub)
                else:
                    _tma_launch(self, x, out, tile_expert, tile_row, tile_n, total,
                                H, T_perm, mt_ub)
                return out
            except Exception:
                _TMA_STATE["level"] += 1
                out = torch.empty(T_perm, self.I, dtype=torch.bfloat16, device=dev)
        _moe_up_kernel[grid](
            x, self.W_gate, self.W_up, out,
            tile_expert, tile_row, tile_n, total,
            H, self.I, mt_ub,
            BM=_BM,
        )
        return out


T_total = 32768
H = 4096
I = 1536  # noqa: E741
E = 128
K = 8


def _build_routing(T_total: int, E: int, K: int, device: str = "cpu") -> torch.Tensor:
    T_perm = T_total * K
    base = T_perm // E
    rem = T_perm - base * E
    counts = torch.full((E,), base, dtype=torch.int32, device=device)
    counts[:rem] += 1
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return offsets


def get_inputs():
    T_perm = T_total * K
    hidden_states = torch.randn(T_perm, H, dtype=torch.bfloat16) * 0.1
    expert_offsets = _build_routing(T_total, E, K)
    return [hidden_states, expert_offsets]


def get_init_inputs():
    return [T_total, H, I, E, K]
