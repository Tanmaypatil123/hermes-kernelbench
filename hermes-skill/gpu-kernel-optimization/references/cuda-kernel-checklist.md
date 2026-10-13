# CUDA C++ kernel checklist

For kernels written in CUDA C++ and built with `torch.utils.cpp_extension.load_inline`. Each item
is a common reason a kernel is wrong, does not build, or runs far below the roofline.

## A `load_inline` template that builds

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// 128-bit vector loads: 4 floats per thread per step, scalar tail for the remainder.
__global__ void scale_add_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                 float* __restrict__ out, float alpha, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t n4 = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* y4 = reinterpret_cast<const float4*>(y);
    float4* o4 = reinterpret_cast<float4*>(out);
    for (int64_t j = i; j < n4; j += stride) {            // grid-stride loop
        float4 a = x4[j], b = y4[j];
        o4[j] = make_float4(alpha * a.x + b.x, alpha * a.y + b.y, alpha * a.z + b.z, alpha * a.w + b.w);
    }
    for (int64_t j = n4 * 4 + i; j < n; j += stride)      // tail
        out[j] = alpha * x[j] + y[j];
}

torch::Tensor scale_add(torch::Tensor x, torch::Tensor y, double alpha) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && y.scalar_type() == torch::kFloat32, "float32 only");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && x.numel() == y.numel(), "contiguous, same size");
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;
    int threads = 256;
    int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int64_t blocks = std::min<int64_t>((n / 4 + threads - 1) / threads + 1, (int64_t)sms * 8);
    auto stream = at::cuda::getCurrentCUDAStream();     // never launch on the default stream
    scale_add_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), (float)alpha, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();                     // surfaces launch errors immediately
    return out;
}
"""
cpp_src = "torch::Tensor scale_add(torch::Tensor x, torch::Tensor y, double alpha);"

major, minor = torch.cuda.get_device_capability()
ext = load_inline(
    name=f"scale_add_v1_sm{major}{minor}",         # bump the version when the source changes
    cpp_sources=cpp_src, cuda_sources=cuda_src, functions=["scale_add"],
    extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
    verbose=False)
```

Notes on the template:
- The function declaration goes in `cpp_sources`; the definition in `cuda_sources`. Python
  receives `double` and `int64_t`, not `float` / `int`.
- `data_ptr<T>()` takes `float`, `at::Half`, `at::BFloat16`, `int64_t`, `int32_t`, `uint8_t`.
  `data_ptr<long long>()` or `data_ptr<long>()` can be an undefined symbol at import time.
- Compile for the GPU you run on (`-gencode` from `get_device_capability()`): building for the wrong
  arch fails at launch with "no kernel image is available".
- `load_inline` needs `ninja` on PATH. A build error appears as an import error in the test output;
  read the full compiler message, which starts at the first `error:` line.

## Memory access

- **Coalesce.** Consecutive threads read consecutive addresses. For a row-major matrix, map
  `threadIdx.x` to the column (fastest-moving) index.
- **Vectorize.** 128-bit loads (`float4`, `int4`, 8 x bf16) cut instruction count. Require 16-byte
  alignment (`reinterpret_cast` on a base pointer from `torch.empty` is aligned; offsets must be
  multiples of 4 floats) and handle the tail with scalar code.
- **Read once.** Fuse every pass over the same data into one kernel; keep a row in registers or
  shared memory instead of re-reading it from DRAM.
- **Mark inputs `const T* __restrict__`** so the compiler can use the read-only path.
- **64-bit indices.** `blockIdx.x * blockDim.x` is 32-bit; cast to `int64_t` before multiplying by a
  row length when tensors exceed 2^31 elements.

## Parallelism and occupancy

- **Fill the GPU.** Launch at least 2-4 waves of blocks (blocks >= 2-4 x SM count x blocks per SM).
  Batch-1 workloads need work split along the long dimension (split-K, segmented rows) plus a small
  second pass to combine the partial results.
- **Block size.** 128-512 threads, a multiple of 32. Use `__launch_bounds__(THREADS)` so the compiler
  can budget registers.
- **Registers vs occupancy.** Large per-thread arrays spill to local memory (slow). Check with
  `-Xptxas -v` in `extra_cuda_cflags`; "spill stores" above 0 in a hot loop is a problem.

## Reductions (max, sum, argmax, top-k)

- **Within a warp:** `__shfl_xor_sync(0xffffffff, v, offset)` for offsets 16, 8, 4, 2, 1; no shared
  memory, no `__syncthreads()`.
- **Within a block:** warp results go to shared memory (one slot per warp), `__syncthreads()`, then
  the first warp reduces them.
- **Across blocks:** write per-block partials and launch a second small kernel; avoid atomics on a
  single address (contention serializes them). Float `atomicAdd` also makes results
  non-deterministic.
- **Argmax / top-k:** reduce `(value, index)` pairs and break ties by the lower index so the result
  is deterministic.

## Shared memory

- 32 banks of 4 bytes: threads of a warp reading the same bank at different addresses serialize.
  Pad 2-D tiles by one element (`tile[32][33]`) for column access.
- Above 48 KB of dynamic shared memory per block, opt in first:
  `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`. The maximum
  depends on the GPU (see `gpu-architectures.md`).
- `__syncthreads()` must be reached by every thread of the block; never put it inside a branch only
  some threads take.

## Tensor cores (matmul-shaped work)

- sm70-sm89 and sm120: `mma.sync` / WMMA fragments (`nvcuda::wmma`), with `ldmatrix` to load tiles.
- sm80+: `cp.async` to stage tiles in shared memory while the previous tile computes.
- sm90 only: `wgmma` + TMA. sm100 only: `tcgen05` + TMEM. Neither exists on sm120.
- Hand-written tensor-core GEMMs rarely beat CUTLASS or Triton `tl.dot`; write CUDA by hand when the
  win is fusion (dequantization, activation, masking), not the MMA itself.

## Numerics

- Accumulate in fp32 even for fp16/bf16 inputs.
- `--use_fast_math` and `__expf` trade accuracy for speed; check the tolerance before using them.
  `expf` of large inputs overflows: subtract the maximum first.
- bf16/fp16 conversions: `__float2bfloat16`, `__bfloat162float` (`#include <cuda_bf16.h>`).

## Debugging and profiling

- `CUDA_LAUNCH_BLOCKING=1` makes an error point at the kernel that caused it.
- `compute-sanitizer --tool memcheck python test.py` finds out-of-bounds accesses that tests miss.
- `ncu --section SpeedOfLight python bench.py` shows achieved DRAM and compute throughput as a
  percentage of peak: the same roofline number the procedure asks for.
