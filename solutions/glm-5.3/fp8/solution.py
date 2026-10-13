"""FP8 e4m3 x e4m3 GEMM via Triton tensor-core tl.dot (fp32 accumulate),
per-output-channel dequant scale, bf16 output.

y = (x @ w.T) * weight_scale,  x:(M,K) fp8, w:(N,K) fp8, scale:(N,) fp32.

When K is not a multiple of 128 the odd row stride defeats vectorized loads
(killing the kernel by ~10x), so we zero-pad x and w along K to a multiple
of 128 -- exact for the dot product.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ["fp8_e4m3"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]
E4M3_MAX = 448.0

PAD_MULT = 128


@triton.jit
def _gemm_kernel(
    X, W, Scale, Y,
    M, N, K,
    sxm, sxk, swn, swk, sym, syn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = X + offs_m[:, None] * sxm + offs_k[None, :] * sxk
    w_ptrs = W + offs_k[:, None] * swk + offs_n[None, :] * swn

    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kmask = offs_k < K - k0
        x = tl.load(x_ptrs, mask=mask_m & kmask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=kmask[:, None] & mask_n, other=0.0)
        acc = tl.dot(x, w, acc, out_dtype=tl.float32)
        x_ptrs += BLOCK_K * sxk
        w_ptrs += BLOCK_K * swk

    scale = tl.load(Scale + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * scale[None, :]
    y_ptrs = Y + offs_m[:, None] * sym + offs_n[None, :] * syn
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_m & mask_n)


def _pick(M, N, K):
    # (BM, BN, BK, GROUP_M, warps, stages)
    if M <= 32:
        return (16, 256, 128, 8, 8, 4)
    if N >= 8192:
        return (128, 256, 64, 8, 8, 4)
    return (128, 256, 128, 8, 8, 3)


def fp8_gemm(x, w, scale):
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    BM, BN, BK, G, nw, ns = _pick(M, N, K)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _gemm_kernel[grid](x, w, scale, y, M, N, K,
                       x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                       y.stride(0), y.stride(1),
                       BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=G,
                       num_warps=nw, num_stages=ns)
    return y


class Model(nn.Module):
    def __init__(self, M: int, N: int, K: int):
        super().__init__()
        self.M, self.N, self.K = M, N, K
        w = torch.empty(N, K, dtype=torch.bfloat16)
        nn.init.normal_(w, std=0.02)
        s = (w.float().abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp(min=1e-12)
        w_fp8 = (w.float() / s).to(torch.float8_e4m3fn)
        self.register_buffer("weight", w_fp8)
        self.register_buffer("weight_scale", s.squeeze(1).to(torch.float32))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        K = self.weight.shape[1]
        if K % PAD_MULT != 0:
            Kp = -(-K // PAD_MULT) * PAD_MULT
            w = self.weight
            wp = torch.zeros((w.shape[0], Kp), dtype=w.dtype, device=w.device)
            wp[:, :K].copy_(w)
            xp = torch.zeros((x.shape[0], Kp), dtype=x.dtype, device=x.device)
            xp[:, :K].copy_(x)
            return fp8_gemm(xp, wp, self.weight_scale)
        return fp8_gemm(x, self.weight, self.weight_scale)



M = 4096
N = 4096
K = 4096


def get_inputs():
    x = (torch.rand(M, K) * 8 - 4).to(torch.float8_e4m3fn)
    return [x]


def get_init_inputs():
    return [M, N, K]
