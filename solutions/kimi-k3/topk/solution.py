"""Custom CUDA top-k (warp bitonic sort + tree merge); selection implemented by hand."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

OP_TYPE = "topk"
SUPPORTED_PRECISIONS = ["fp32"]
HARDWARE_REQUIRED = ["RTX_PRO_6000", "H100", "B200"]

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>
#include <mutex>

#define NWARPS 8
#define NTHREADS (NWARPS * 32)
#define KMAX 64
#define MAXSLICES 64
#define WSLISTS 64

// ---------------- scalar-layout path: element i = lane + 32*r ----------------

__device__ __forceinline__ void sort128_desc_s(float v[4], int id[4], int lane) {
    for (int size = 2; size <= 128; size <<= 1) {
        for (int j = size >> 1; j > 0; j >>= 1) {
            if (j >= 32) {
                int rb = j >> 5;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    if (!(r & rb)) {
                        int r2 = r | rb;
                        int i = lane + 32 * r;
                        bool desc = (i & size) == 0;
                        bool sw = desc ? (v[r] < v[r2]) : (v[r] > v[r2]);
                        if (sw) {
                            float tv = v[r]; v[r] = v[r2]; v[r2] = tv;
                            int ti = id[r]; id[r] = id[r2]; id[r2] = ti;
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int i = lane + 32 * r;
                    bool desc = (i & size) == 0;
                    float ov = __shfl_xor_sync(0xffffffffu, v[r], j);
                    int   oi = __shfl_xor_sync(0xffffffffu, id[r], j);
                    bool lower = (lane & j) == 0;
                    bool take = (desc == lower) ? (ov > v[r]) : (ov < v[r]);
                    if (take) { v[r] = ov; id[r] = oi; }
                }
            }
        }
    }
}

// ---------------- vectorized path: element i = 4*lane + r -------------------

__device__ __forceinline__ void sort128_desc_v(float v[4], int id[4], int lane) {
    for (int size = 2; size <= 128; size <<= 1) {
        for (int j = size >> 1; j > 0; j >>= 1) {
            if (j >= 4) {
                int lb = j >> 2;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int i = 4 * lane + r;
                    bool desc = (i & size) == 0;
                    float ov = __shfl_xor_sync(0xffffffffu, v[r], lb);
                    int   oi = __shfl_xor_sync(0xffffffffu, id[r], lb);
                    bool lower = (lane & lb) == 0;
                    bool take = (desc == lower) ? (ov > v[r]) : (ov < v[r]);
                    if (take) { v[r] = ov; id[r] = oi; }
                }
            } else {
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    if (!(r & j)) {
                        int r2 = r | j;
                        int i = 4 * lane + r;
                        bool desc = (i & size) == 0;
                        bool sw = desc ? (v[r] < v[r2]) : (v[r] > v[r2]);
                        if (sw) {
                            float tv = v[r]; v[r] = v[r2]; v[r2] = tv;
                            int ti = id[r]; id[r] = id[r2]; id[r2] = ti;
                        }
                    }
                }
            }
        }
    }
}

// redistribute from layout i = 4*lane + r to layout i = lane + 32*r (top 64)
__device__ __forceinline__ void redistribute_top64(float v[2], int id[2],
                                                   float sv[4], int sid[4], int lane) {
    #pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        int i = lane + 32 * rr;      // target position, 0..63
        int src = i >> 2;            // source lane
        int sr  = i & 3;             // source register
        float c0 = __shfl_sync(0xffffffffu, sv[0], src);
        float c1 = __shfl_sync(0xffffffffu, sv[1], src);
        float c2 = __shfl_sync(0xffffffffu, sv[2], src);
        float c3 = __shfl_sync(0xffffffffu, sv[3], src);
        int d0 = __shfl_sync(0xffffffffu, sid[0], src);
        int d1 = __shfl_sync(0xffffffffu, sid[1], src);
        int d2 = __shfl_sync(0xffffffffu, sid[2], src);
        int d3 = __shfl_sync(0xffffffffu, sid[3], src);
        v[rr]  = sr == 0 ? c0 : (sr == 1 ? c1 : (sr == 2 ? c2 : c3));
        id[rr] = sr == 0 ? d0 : (sr == 1 ? d1 : (sr == 2 ? d2 : d3));
    }
}

// ---------------- shared merge primitives (layout i = lane + 32*r) ----------

__device__ __forceinline__ void bitonic_merge64_desc(float v[2], int id[2], int lane) {
    if (v[0] < v[1]) {
        float tv = v[0]; v[0] = v[1]; v[1] = tv;
        int ti = id[0]; id[0] = id[1]; id[1] = ti;
    }
    for (int j = 16; j > 0; j >>= 1) {
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            float ov = __shfl_xor_sync(0xffffffffu, v[r], j);
            int   oi = __shfl_xor_sync(0xffffffffu, id[r], j);
            bool lower = (lane & j) == 0;
            bool take = lower ? (ov > v[r]) : (ov < v[r]);
            if (take) { v[r] = ov; id[r] = oi; }
        }
    }
}

__device__ __forceinline__ void merge64_keep(float av[2], int ai[2],
                                             const float bv[2], const int bi[2], int lane) {
    float bh_v = __shfl_xor_sync(0xffffffffu, bv[1], 31); // b[63 - lane]
    int   bh_i = __shfl_xor_sync(0xffffffffu, bi[1], 31);
    float bl_v = __shfl_xor_sync(0xffffffffu, bv[0], 31); // b[31 - lane]
    int   bl_i = __shfl_xor_sync(0xffffffffu, bi[0], 31);
    if (bh_v > av[0]) { av[0] = bh_v; ai[0] = bh_i; }
    if (bl_v > av[1]) { av[1] = bl_v; ai[1] = bl_i; }
    bitonic_merge64_desc(av, ai, lane);
}

// tree merge of sorted lists in smem; each warp takes up to SLOTS merges/round
template <int SLOTS>
__device__ __forceinline__ void tree_merge_smem(float* sv, int* si, int m0,
                                                int warp, int lane) {
    for (int m = m0; m > 1; m >>= 1) {
        int half = m >> 1;
        float xv[SLOTS][2]; int xi[SLOTS][2]; bool act[SLOTS];
        #pragma unroll
        for (int s = 0; s < SLOTS; s++) {
            int idx = warp + s * NWARPS;
            act[s] = idx < half;
            if (act[s]) {
                const float* a = sv + (2 * idx) * WSLISTS;
                const float* b = a + WSLISTS;
                const int*   ap = si + (2 * idx) * WSLISTS;
                const int*   bp = ap + WSLISTS;
                float av[2] = {a[lane], a[lane + 32]};
                int   ai[2] = {ap[lane], ap[lane + 32]};
                float bv[2] = {b[lane], b[lane + 32]};
                int   bi[2] = {bp[lane], bp[lane + 32]};
                merge64_keep(av, ai, bv, bi, lane);
                xv[s][0] = av[0]; xv[s][1] = av[1];
                xi[s][0] = ai[0]; xi[s][1] = ai[1];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int s = 0; s < SLOTS; s++) {
            int idx = warp + s * NWARPS;
            if (act[s]) {
                float* o = sv + idx * WSLISTS;
                int*   op = si + idx * WSLISTS;
                o[lane] = xv[s][0]; o[lane + 32] = xv[s][1];
                op[lane] = xi[s][0]; op[lane + 32] = xi[s][1];
            }
        }
        __syncthreads();
    }
}

template <bool VEC>
__global__ void topk_fused(const float* __restrict__ x,
                           float* __restrict__ ws_v, int* __restrict__ ws_i,
                           unsigned int* __restrict__ cnt,
                           float* __restrict__ out_v, long* __restrict__ out_i,
                           int n, int k, int slices, int slice_len, int mpad) {
    __shared__ float sv[MAXSLICES * WSLISTS];
    __shared__ int   si[MAXSLICES * WSLISTS];
    __shared__ bool amlast;

    int slice = blockIdx.x % slices;
    int row   = blockIdx.x / slices;
    int s0 = slice * slice_len;
    int s1 = min(n, s0 + slice_len);

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    float rv[2] = {-CUDART_INF_F, -CUDART_INF_F};
    int   ri[2] = {-1, -1};

    int nchunk = (s1 - s0 + 127) >> 7;
    const float* xrow = x + (size_t)row * n;
    for (int c = warp; c < nchunk; c += NWARPS) {
        int base = s0 + (c << 7);
        float bv[2]; int bi[2];
        if (VEC) {
            const float4* x4 = reinterpret_cast<const float4*>(xrow);
            int q0 = base >> 2, q1 = s1 >> 2;
            float v[4]; int id[4];
            float4 f = (q0 + lane < q1) ? x4[q0 + lane]
                                        : make_float4(-CUDART_INF_F, -CUDART_INF_F,
                                                      -CUDART_INF_F, -CUDART_INF_F);
            int e = base + 4 * lane;
            v[0] = f.x; v[1] = f.y; v[2] = f.z; v[3] = f.w;
            id[0] = e; id[1] = e + 1; id[2] = e + 2; id[3] = e + 3;
            sort128_desc_v(v, id, lane);
            redistribute_top64(bv, bi, v, id, lane);
        } else {
            float v[4]; int id[4];
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                int e = base + lane + 32 * r;
                if (e < s1) { v[r] = xrow[e]; id[r] = e; }
                else        { v[r] = -CUDART_INF_F; id[r] = -1; }
            }
            sort128_desc_s(v, id, lane);
            bv[0] = v[0]; bv[1] = v[1];
            bi[0] = id[0]; bi[1] = id[1];
        }
        merge64_keep(rv, ri, bv, bi, lane);
    }

    sv[warp * WSLISTS + lane]      = rv[0];
    sv[warp * WSLISTS + lane + 32] = rv[1];
    si[warp * WSLISTS + lane]      = ri[0];
    si[warp * WSLISTS + lane + 32] = ri[1];
    __syncthreads();

    tree_merge_smem<1>(sv, si, NWARPS, warp, lane);

    size_t gbase = ((size_t)row * slices + slice) * WSLISTS;
    for (int i = threadIdx.x; i < WSLISTS; i += blockDim.x) {
        ws_v[gbase + i] = sv[i];
        ws_i[gbase + i] = si[i];
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        // last block for the row; reset counter so the next call starts at 0
        amlast = (atomicAdd(&cnt[row], 1u) % (unsigned)slices == (unsigned)(slices - 1));
        if (amlast) cnt[row] = 0;
    }
    __syncthreads();
    if (!amlast) return;

    for (int i = threadIdx.x; i < mpad * WSLISTS; i += blockDim.x) {
        bool valid = i < slices * WSLISTS;
        sv[i] = valid ? ws_v[(size_t)row * slices * WSLISTS + i] : -CUDART_INF_F;
        si[i] = valid ? ws_i[(size_t)row * slices * WSLISTS + i] : -1;
    }
    __syncthreads();

    tree_merge_smem<4>(sv, si, mpad, warp, lane);

    if (threadIdx.x < k) {
        size_t off = (size_t)row * k + threadIdx.x;
        out_v[off] = sv[threadIdx.x];
        out_i[off] = (long)si[threadIdx.x];
    }
}

static torch::Tensor g_ws_v, g_ws_i, g_cnt;
static std::mutex g_mu;

std::vector<torch::Tensor> topk_cuda(torch::Tensor x, int64_t k) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kFloat,
                "expected 2D fp32 CUDA tensor");
    TORCH_CHECK(k >= 1 && k <= KMAX, "k must be in [1, 64]");
    auto xc = x.contiguous();
    int64_t batch = xc.size(0), n = xc.size(1);
    TORCH_CHECK(k <= n, "k must be <= n");

    auto vals = torch::empty({batch, k}, xc.options());
    auto idx  = torch::empty({batch, k}, xc.options().dtype(torch::kLong));

    int occ_slices   = (int)((600 + batch - 1) / batch);
    int size_slices  = (int)((n + 255) / 256);
    int chunk_slices = (int)((n + 127) / 128);
    int slices = std::max(1, std::min(std::min(occ_slices, MAXSLICES),
                                      std::min(size_slices, chunk_slices)));
    int slice_len = (int)((((n + slices - 1) / slices) + 127) / 128 * 128);
    slices = (int)((n + slice_len - 1) / slice_len);
    int mpad = 1; while (mpad < slices) mpad <<= 1;

    auto stream = at::cuda::getCurrentCUDAStream();
    {
        std::lock_guard<std::mutex> lock(g_mu);
        int64_t need = batch * slices * WSLISTS;
        if (!g_ws_v.defined() || g_ws_v.numel() < need || g_ws_v.device() != xc.device()) {
            g_ws_v = torch::empty({need}, xc.options());
            g_ws_i = torch::empty({need}, xc.options().dtype(torch::kInt));
        }
        if (!g_cnt.defined() || g_cnt.numel() < batch || g_cnt.device() != xc.device()) {
            g_cnt = torch::zeros({batch}, xc.options().dtype(torch::kInt));
        }
    }

    bool vec = (n % 4 == 0);
    if (vec) {
        topk_fused<true><<<(unsigned)(batch * slices), NTHREADS, 0, stream>>>(
            xc.data_ptr<float>(), g_ws_v.data_ptr<float>(), g_ws_i.data_ptr<int>(),
            (unsigned int*)g_cnt.data_ptr<int>(),
            vals.data_ptr<float>(), (long*)idx.data_ptr<int64_t>(),
            (int)n, (int)k, slices, slice_len, mpad);
    } else {
        topk_fused<false><<<(unsigned)(batch * slices), NTHREADS, 0, stream>>>(
            xc.data_ptr<float>(), g_ws_v.data_ptr<float>(), g_ws_i.data_ptr<int>(),
            (unsigned int*)g_cnt.data_ptr<int>(),
            vals.data_ptr<float>(), (long*)idx.data_ptr<int64_t>(),
            (int)n, (int)k, slices, slice_len, mpad);
    }

    return {vals, idx};
}
"""

cpp_src = "std::vector<torch::Tensor> topk_cuda(torch::Tensor x, int64_t k);"

_mod = load_inline(
    name="topk_bitonic_ext_v4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["topk_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, batch: int, n: int, k: int):
        super().__init__()
        self.batch, self.n, self.k = batch, n, k
        self.register_buffer("_dummy", torch.zeros(1))
        self._graph = None
        self._in = None
        self._ov = None
        self._oi = None

    def _direct(self, x):
        return _mod.topk_cuda(x, self.k)

    def forward(self, x: torch.Tensor):
        return _mod.topk_cuda(x, self.k)


batch = 64
n = 8192
k = 8


def get_inputs():
    x = torch.randn(batch, n, dtype=torch.float32)
    return [x]


def get_init_inputs():
    return [batch, n, k]
