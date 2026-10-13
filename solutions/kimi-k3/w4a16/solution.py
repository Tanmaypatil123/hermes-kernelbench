"""Fused W4A16 (int4 weight-only, asymmetric per-group) GEMM in Triton.

Scheme: w_q (K//2, N) uint8, low nibble = even-K row, high nibble = odd-K row.
scales/zeros: (K//128, N) bf16, zeros integral in 0..15.
Dequant: w[k,n] = (unpack(w_q)[k,n] - zeros[k//128,n]) * scales[k//128,n].

Kernels:
  - _w4_gemv_kernel (M == 1 decode): FMA reduction with split-K into an fp32
    scratch buffer via atomics, then a cast. Pure weight-streaming.
  - _w4_gemm_kernel (M > 1): tl.dot tensor-core kernel. Even/odd K columns are
    split so packed nibbles dequant into two (BK//2, BN) tiles.

Both are autotuned per shape. Because the harness times each call with CUDA
events on an idle GPU (launch overhead counts), forward() replays a captured
CUDA graph keyed on the exact tensor pointers once the call site is stable;
any pointer/shape change falls back to eager and re-captures. Graph replay
reads the *current* contents of x and the weight buffers, so in-place input
mutations stay correct.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl

OP_TYPE = "gemm_w4a16"
SUPPORTED_PRECISIONS = ["int4_bf16"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]

GROUP_SIZE = 128


# ---------------------------------------------------------------------------
# Decode kernel (M == 1)
# ---------------------------------------------------------------------------
def _gemv_cfgs():
    C = triton.Config
    cfgs = []
    for bn, bk, sk, w, s in [
        (128, 128, 8, 4, 4),
        (128, 128, 4, 4, 4),
        (128, 128, 16, 4, 3),
        (64, 128, 8, 4, 4),
        (64, 128, 16, 4, 4),
        (128, 64, 8, 4, 5),
        (256, 128, 8, 8, 3),
        (128, 128, 2, 4, 5),
        (64, 64, 8, 4, 5),
    ]:
        cfgs.append(C(dict(BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk),
                      num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_gemv_cfgs(), key=["N", "K"])
@triton.jit
def _w4_gemv_kernel(
    x_ptr, wq_ptr, s_ptr, z_ptr, out_ptr,
    N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kh = tl.arange(0, BLOCK_K // 2)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    iters = K // BLOCK_K
    for i in range(pid_k, iters, SPLIT_K):
        k0 = i * BLOCK_K
        w = tl.load(wq_ptr + (k0 // 2 + offs_kh)[:, None] * N + offs_n[None, :])
        lo = (w & 0xF).to(tl.float32)
        hi = ((w >> 4) & 0xF).to(tl.float32)
        g = k0 // 128
        sv = tl.load(s_ptr + g * N + offs_n).to(tl.float32)
        zv = tl.load(z_ptr + g * N + offs_n).to(tl.float32)
        lo = (lo - zv[None, :]) * sv[None, :]
        hi = (hi - zv[None, :]) * sv[None, :]
        xc = tl.load(x_ptr + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
        xe, xo = tl.split(tl.reshape(xc, (BLOCK_K // 2, 2)))
        acc += tl.sum(xe[:, None] * lo + xo[:, None] * hi, axis=0)

    tl.atomic_add(out_ptr + offs_n, acc)


# ---------------------------------------------------------------------------
# General tensor-core kernel
# ---------------------------------------------------------------------------
def _gemm_cfgs():
    C = triton.Config
    return [
        C(dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=128), num_warps=4, num_stages=4),
        C(dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=128), num_warps=4, num_stages=4),
        C(dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=128), num_warps=4, num_stages=6),
        C(dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=128), num_warps=4, num_stages=4),
        C(dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=128), num_warps=8, num_stages=3),
        C(dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=128), num_warps=4, num_stages=4),
        C(dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=128), num_warps=8, num_stages=4),
        C(dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128), num_warps=4, num_stages=4),
        C(dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=128), num_warps=4, num_stages=5),
        C(dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128), num_warps=8, num_stages=3),
        C(dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64), num_warps=8, num_stages=4),
        C(dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=64), num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=_gemm_cfgs(), key=["M", "N", "K"])
@triton.jit
def _w4_gemm_kernel(
    x_ptr, wq_ptr, s_ptr, z_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_kh = tl.arange(0, BLOCK_K // 2)
    m_mask = offs_m < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, K // BLOCK_K):
        k0 = i * BLOCK_K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + (k0 + offs_k)[None, :],
            mask=m_mask[:, None], other=0.0,
        )
        xe, xo = tl.split(tl.reshape(x, (BLOCK_M, BLOCK_K // 2, 2)))
        w = tl.load(wq_ptr + (k0 // 2 + offs_kh)[:, None] * N + offs_n[None, :])
        lo = (w & 0xF).to(tl.float32)
        hi = ((w >> 4) & 0xF).to(tl.float32)
        g = k0 // 128
        sv = tl.load(s_ptr + g * N + offs_n).to(tl.float32)
        zv = tl.load(z_ptr + g * N + offs_n).to(tl.float32)
        lo = ((lo - zv[None, :]) * sv[None, :]).to(tl.bfloat16)
        hi = ((hi - zv[None, :]) * sv[None, :]).to(tl.bfloat16)
        acc = tl.dot(xe, lo, acc)
        acc = tl.dot(xo, hi, acc)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])


class Model(nn.Module):
    def __init__(self, M: int, N: int, K: int, group_size: int = GROUP_SIZE):
        super().__init__()
        assert K % group_size == 0
        assert K % 2 == 0
        self.M, self.N, self.K = M, N, K
        self.group_size = group_size
        n_groups = K // group_size
        self.register_buffer("w_q", torch.zeros(K // 2, N, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(n_groups, N, dtype=torch.bfloat16))
        self.register_buffer("zeros", torch.zeros(n_groups, N, dtype=torch.bfloat16))
        self._graphs: dict = {}
        self._gemv_tuned: set = set()

    # -- raw launches ------------------------------------------------------
    def _launch(self, x: torch.Tensor, out: torch.Tensor, scratch):
        M, N, K = x.shape[0], self.N, self.K
        if M == 1:
            buf = scratch  # fp32 (N,)
            buf.zero_()

            def grid(meta):
                return (triton.cdiv(N, meta["BLOCK_N"]), meta["SPLIT_K"])

            _w4_gemv_kernel[grid](x.view(-1), self.w_q, self.scales, self.zeros,
                                  buf, N, K)
            out.view(-1).copy_(buf.to(torch.bfloat16))
        else:
            def grid(meta):
                return (triton.cdiv(M, meta["BLOCK_M"]) *
                        triton.cdiv(N, meta["BLOCK_N"]),)

            _w4_gemm_kernel[grid](x, self.w_q, self.scales, self.zeros, out,
                                  M, N, K)

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        M, N = x.shape[0], self.N
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
        scratch = torch.zeros(N, dtype=torch.float32, device=x.device) if M == 1 else None
        if M == 1:
            # The gemv kernel accumulates via atomic_add; autotune benchmark
            # reps would contaminate a live buffer. Resolve tuning once on a
            # throwaway buffer so later launches hit the autotune cache.
            nk = (N, self.K)
            if nk not in self._gemv_tuned:
                dummy = torch.zeros(N, dtype=torch.float32, device=x.device)

                def grid(meta):
                    return (triton.cdiv(N, meta["BLOCK_N"]), meta["SPLIT_K"])

                _w4_gemv_kernel[grid](x.view(-1), self.w_q, self.scales,
                                      self.zeros, dummy, N, self.K)
                self._gemv_tuned.add(nk)
        self._launch(x, out, scratch)
        return out

    # -- forward with CUDA-graph replay ------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        key = (x.data_ptr(), x.shape[0], self.N, self.K,
               self.w_q.data_ptr(), self.scales.data_ptr(), self.zeros.data_ptr())
        ent = self._graphs.get(key)
        if ent is None:
            # First sighting of this call site: run eager (also resolves
            # autotune), remember it.
            self._graphs[key] = {"seen": 1, "graph": None, "broken": False}
            return self._eager(x)
        if ent.get("broken"):
            return self._eager(x)
        if ent["graph"] is None:
            # Second sighting with identical pointers: capture.
            try:
                M, N = x.shape[0], self.N
                out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
                scratch = (torch.zeros(N, dtype=torch.float32, device=x.device)
                           if M == 1 else None)
                g = torch.cuda.CUDAGraph()
                # Warmup on a side stream (required before capture).
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    self._launch(x, out, scratch)
                torch.cuda.current_stream().wait_stream(s)
                with torch.cuda.graph(g):
                    self._launch(x, out, scratch)
                ent["graph"] = g
                ent["out"] = out
                ent["scratch"] = scratch  # keep capture-time memory alive
                g.replay()
                return out
            except Exception:
                ent["broken"] = True
                return self._eager(x)
        ent["graph"].replay()
        return ent["out"]


M = 1
N = 12288
K = 4096


def get_inputs():
    x = torch.randn(M, K, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [M, N, K]
