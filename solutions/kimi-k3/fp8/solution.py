"""FP8 e4m3 GEMM (Triton, fp8 tensor-core MMA with fp32 accumulate).

y = ((x @ w.T) * weight_scale).to(bf16)
x: fp8_e4m3 (M, K); weight: fp8_e4m3 (N, K); weight_scale: (N,) fp32.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

OP_TYPE = "gemm"
SUPPORTED_PRECISIONS = ["fp8_e4m3"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]
E4M3_MAX = 448.0


@triton.jit
def _fp8_gemm_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
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
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < k_rem), other=0.0)
        b = tl.load(w_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < k_rem), other=0.0)
        acc = tl.dot(a, tl.trans(b), acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    s = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * s[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def _launch(x, w, s, y):
    M, K = x.shape
    N = w.shape[0]
    if M <= 64:
        BLOCK_M, BLOCK_N, BLOCK_K = 32 if M <= 32 else 64, 64, 256
        num_warps, num_stages = 4, 4
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
        num_warps, num_stages = 8, 4
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _fp8_gemm_kernel[grid](
        x, w, s, y, M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=8,
        num_warps=num_warps, num_stages=num_stages,
    )


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
        M, K = x.shape
        w = self.weight
        # fp8 loads need K aligned; pad K up to a multiple of 128 when odd.
        if K % 128 != 0:
            Kp = (K + 127) & ~127
            xp = torch.zeros((M, Kp), device=x.device, dtype=x.dtype)
            xp[:, :K] = x
            wp = torch.zeros((self.N, Kp), device=x.device, dtype=x.dtype)
            wp[:, :K] = w
            x, w = xp, wp
        y = torch.empty((M, self.N), device=x.device, dtype=torch.bfloat16)
        _launch(x, w, self.weight_scale, y)
        return y


M = 4096
N = 4096
K = 4096


def get_inputs():
    x = (torch.rand(M, K) * 8 - 4).to(torch.float8_e4m3fn)
    return [x]


def get_init_inputs():
    return [M, N, K]
