"""FP8 e4m3 x e4m3 GEMM for RTX PRO 6000 (SM120 Blackwell).

y = ((x @ w.T) * weight_scale).to(bf16)

Real fp8 tensor-core path: Triton tl.dot on float8_e4m3fn operands with fp32
accumulate (compiles to mma.sync ... e4m3 on SM120), per-output-channel
dequant scale applied in the epilogue.

Launch paths:
  - M > 64: classic swizzled tiled GEMM (autotuned over a small config set).
    Unaligned K (e.g. 4127) is zero-padded to a 128-multiple (cached weight
    pad, reused activation pad buffer) to keep loads 16B-aligned/vectorized.
  - M <= 64 (decode / skinny): single-launch memory-bound kernel, one CTA per
    N tile, full K loop, direct scaled bf16 store (no partials, no reduce).
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

E4M3_MAX = 448.0


def _gemm_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 4}, num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=_gemm_configs(), key=['M', 'N', 'K'])
@triton.jit
def _fp8_gemm_kernel(A, B, C, WScale, M, N, K,
                     stride_am, stride_ak, stride_bn, stride_bk,
                     stride_cm, stride_cn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # wrap row indices so A/B loads never need OOB masks (stores are masked)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)
    a_ptrs = A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    wscale = tl.load(WScale + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * wscale[None, :]
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask)


def _tma_alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


_TMA_READY = False
try:
    triton.set_allocator(_tma_alloc_fn)
    _TMA_READY = True
except Exception:
    _TMA_READY = False


@triton.jit
def _fp8_gemm_skinny_tma_kernel(A, B, C, WScale, M, N, K,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                BLOCK_K: tl.constexpr):
    """TMA bulk-load variant of the skinny path: no per-element masks, deep
    async pipelining, OOB rows zero-filled by hardware."""
    pid_n = tl.program_id(0)
    a_desc = tl.make_tensor_descriptor(A, shape=[M, K], strides=[K, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(B, shape=[N, K], strides=[K, 1],
                                       block_shape=[BLOCK_N, BLOCK_K])
    n0 = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([0, kb * BLOCK_K])
        b = b_desc.load([n0, kb * BLOCK_K])
        acc = tl.dot(a, tl.trans(b), acc)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    ws = tl.load(WScale + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * ws[None, :]
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Measured on the target GPU: the plain skinny kernel (~85.6us on
# M=32/N=8192/K=8192) edges out the TMA variant (~88us), so skip the TMA probe.
_SKINNY_MODE = 'plain'  # None -> probe, 'tma', or 'plain'


@triton.jit
def _fp8_gemm_skinny_kernel(A, B, C, WScale, M, N, K,
                            stride_am, stride_ak, stride_bn, stride_bk,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr):
    """Single-launch memory-bound path for M <= 64: one CTA per N tile,
    full K loop, direct scaled bf16 store. No partials, no reduce kernel."""
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    a_mask = offs_m[:, None] < M
    b_mask = offs_n[:, None] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - kb * BLOCK_K
        a = tl.load(a_ptrs, mask=a_mask & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=b_mask & (offs_k[None, :] < k_remaining), other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    ws = tl.load(WScale + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * ws[None, :]
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=a_mask & (offs_n[None, :] < N))


def _fp8_scaled_gemm(x: torch.Tensor, w: torch.Tensor, wscale: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    if M <= 64:
        global _SKINNY_MODE
        BLOCK_M = max(16, triton.next_power_of_2(M))
        BLOCK_N = 64
        grid_n = triton.cdiv(N, BLOCK_N)
        if _SKINNY_MODE is None and _TMA_READY and (K % 128 == 0) and M >= 16:
            # probe the TMA path once; any compile/launch failure falls back
            try:
                _fp8_gemm_skinny_tma_kernel[(grid_n,)](
                    x, w, y, wscale, M, N, K,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=128,
                    num_warps=4, num_stages=6,
                )
                torch.cuda.synchronize()
                # one-time numerical validation of the TMA path
                ref = (x.float() @ w.float().T) * wscale[None, :]
                err = (y.float() - ref).abs().max().item()
                if not torch.isfinite(y.float()).all().item() or err > 0.05 * ref.abs().max().item() + 0.1:
                    raise RuntimeError(f"tma probe sanity failed: err={err}")
                _SKINNY_MODE = 'tma'
                return y
            except Exception:
                _SKINNY_MODE = 'plain'
        elif _SKINNY_MODE == 'tma' and (K % 128 == 0) and M >= 16:
            _fp8_gemm_skinny_tma_kernel[(grid_n,)](
                x, w, y, wscale, M, N, K,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=128,
                num_warps=4, num_stages=6,
            )
            return y
        # plain fallback
        if BLOCK_M <= 32:
            BLOCK_K, num_stages = 256, 3
        else:
            BLOCK_K, num_stages = 128, 4
        _fp8_gemm_skinny_kernel[(grid_n,)](
            x, w, y, wscale, M, N, K,
            x.stride(0), x.stride(1), w.stride(0), w.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=num_stages,
        )
        return y

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    _fp8_gemm_kernel[grid](
        x, w, y, wscale, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


class Model(nn.Module):
    """y = ((x @ w.T) * weight_scale).to(bf16).

    x: fp8_e4m3 (M, K).  w: fp8_e4m3 (N, K) normalized to the e4m3 range.
    weight_scale: (N,) per-output-channel dequant scale.
    """

    def __init__(self, M: int, N: int, K: int):
        super().__init__()
        self.M, self.N, self.K = M, N, K
        w = torch.empty(N, K, dtype=torch.bfloat16)
        nn.init.normal_(w, std=0.02)
        s = (w.float().abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp(min=1e-12)  # (N,1)
        w_fp8 = (w.float() / s).to(torch.float8_e4m3fn)
        self.register_buffer("weight", w_fp8)                         # (N, K) fp8
        self.register_buffer("weight_scale", s.squeeze(1).to(torch.float32))  # (N,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = x.shape[1]
        if K % 128 == 0:
            return _fp8_scaled_gemm(x, self.weight, self.weight_scale)
        # Unaligned K (e.g. 4127) breaks row alignment and kills load
        # vectorization. Zero-pad both operands to a 128-multiple K; padded
        # lanes contribute exactly 0 to the fp32 accumulate.
        Kp = ((K + 127) // 128) * 128
        w_pad = self._padded_weight(Kp)
        x_pad = self._padded_x(x, Kp)
        return _fp8_scaled_gemm(x_pad, w_pad, self.weight_scale)

    def _padded_x(self, x: torch.Tensor, Kp: int) -> torch.Tensor:
        M, K = x.shape
        key = (M, K, Kp, x.device.index)
        cache = getattr(self, "_x_pad_cache", None)
        if cache is None or cache[0] != key:
            xp = torch.zeros(M, Kp, dtype=self.weight.dtype, device=x.device)
            self._x_pad_cache = (key, xp)
        else:
            xp = cache[1]
        xp[:, :K] = x
        return xp

    def _padded_weight(self, Kp: int) -> torch.Tensor:
        w = self.weight
        key = (w.data_ptr(), w._version, w.shape[0], w.shape[1], Kp)
        cached = getattr(self, "_w_pad_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        wp = torch.zeros(w.shape[0], Kp, dtype=w.dtype, device=w.device)
        wp[:, : w.shape[1]] = w
        self._w_pad_cache = (key, wp)
        return wp


M = 4096
N = 4096
K = 4096


def get_inputs():
    x = (torch.rand(M, K) * 8 - 4).to(torch.float8_e4m3fn)
    return [x]


def get_init_inputs():
    return [M, N, K]
