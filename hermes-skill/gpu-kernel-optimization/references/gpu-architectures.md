# GPU architectures for kernel writers

Run `python3 ${HERMES_SKILL_DIR}/scripts/gpu_info.py` on the target machine (or inside the task's
remote test) for the exact numbers of the GPU you have. The tables below are for planning; when
they disagree with `gpu_info.py`, trust `gpu_info.py`.

## The numbers that decide a design

| GPU | Arch (sm) | SMs | DRAM GB/s | L2 | Max shared mem per block | Tensor-core dtypes |
|---|---|---|---|---|---|---|
| T4 | Turing (75) | 40 | 320 | 4 MB | 64 KB | fp16, int8, int4 |
| A10 / A10G | Ampere (86) | 72 / 80 | 600 | 6 MB | 99 KB | fp16, bf16, tf32, int8 |
| A100 40/80 GB | Ampere (80) | 108 | 1555 / 2039 | 40 MB | 163 KB | fp16, bf16, tf32, int8, fp64 |
| L4 | Ada (89) | 58 | 300 | 48 MB | 99 KB | + fp8 |
| L40S | Ada (89) | 142 | 864 | 96 MB | 99 KB | + fp8 |
| H100 SXM / PCIe | Hopper (90) | 132 / 114 | 3350 / 2000 | 50 MB | 227 KB | + fp8 (wgmma) |
| H200 | Hopper (90) | 132 | 4800 | 50 MB | 227 KB | + fp8 (wgmma) |
| B200 | Blackwell DC (100) | 148 | ~8000 | large | 227 KB | + fp6, fp4 (tcgen05) |
| RTX PRO 6000 / RTX 5090 | Blackwell consumer (120) | 188 / 170 | 1792 / 1792 | large | 99 KB | + fp8, fp6, fp4 (mma.sync) |

Peak FLOP/s are far above what memory can feed for most non-GEMM kernels, so for elementwise ops,
reductions, norms, softmax, top-k, decode-time GEMV and attention decode, **DRAM bandwidth is the
roofline**. Only large GEMMs, prefill attention and grouped GEMMs are compute-bound.

## What each generation changes

- **Turing (sm75, T4).** No bf16 tensor cores: bf16 `tl.dot` or `mma` fails or falls back, so use
  fp16 or fp32. No `cp.async`. Small L2 (4 MB) means almost nothing stays cached between kernels:
  fusing passes is worth even more. 40 SMs fill quickly; small grids are fine.
- **Ampere (sm80, sm86).** bf16 and tf32 tensor cores; `cp.async` lets loads overlap compute
  (Triton uses it automatically via `num_stages`). sm86 has much less shared memory than sm80
  (99 KB vs 163 KB per block): tile sizes tuned on an A100 may not launch on an A10.
- **Ada (sm89, L4, L40S, RTX 40xx).** fp8 tensor cores via `mma.sync`. Large L2 (48-96 MB): a
  tensor that fits in L2 may be re-read nearly for free, so benchmarks must flush L2 to be honest.
  L4 has only 300 GB/s: memory-bound kernels are slow in absolute terms; judge by % of roofline.
- **Hopper (sm90, H100, H200).** Adds TMA (bulk async copies), `wgmma` (warpgroup MMA), thread-block
  clusters and distributed shared memory. Peak GEMM speed needs wgmma + TMA pipelines (CUTLASS 3,
  or Triton, which emits them for `tl.dot` on sm90). `mma.sync` still works but caps around half of
  peak. 227 KB shared memory per block allows deep pipelines.
- **Blackwell datacenter (sm100, B200).** `tcgen05` MMA with tensor memory (TMEM), 2-SM MMA, fp4 and
  fp6. Hopper `wgmma` code does not run here; use CUTLASS 3.8+ / CuTe or a recent Triton.
- **Blackwell consumer (sm120, RTX PRO 6000, RTX 50xx).** Not the same as sm100: **no tcgen05, no
  TMEM, no wgmma**. Tensor cores are driven by `mma.sync` (extended with fp8/fp6/fp4 block-scaled
  formats), shared memory is 99 KB per block, and it has TMA. Code written for H100 or B200 tensor
  instructions will not compile; target `sm_120` / `compute_120` explicitly. High bandwidth
  (1.8 TB/s) with many SMs (170-188): memory-bound kernels need grids of at least several thousand
  blocks, or split work (split-K, split-N, segmented rows), to reach the roofline.

## Grid sizing rule of thumb

Aim for at least 2-4 waves: blocks >= 2-4 x (SM count x resident blocks per SM). A batch-1 decode
GEMV with N = 4096 and 128 columns per block launches 32 blocks: under 20% of a 188-SM GPU. Split
the K dimension (and reduce the partials) or use smaller column tiles until the grid fills the GPU.

## Choosing the tool

| Situation | Use |
|---|---|
| Elementwise, reductions, norms, softmax, fused epilogues | Triton (portable across sm75-sm120) |
| Need warp shuffles, custom shared-memory layouts, bit tricks (top-k, sorting, int4 unpack) | CUDA C++ via `load_inline`, or Triton if it expresses it |
| Peak GEMM / attention on sm90 or sm100 | CUTLASS 3 / CuTe, or Triton `tl.dot` (it emits wgmma/tcgen05) |
| Peak GEMM on sm120 | CUTLASS with sm120 kernels, or Triton `tl.dot` (mma.sync path) |
| Turing (T4) tensor cores | fp16 only; Triton `tl.dot` with fp16 inputs |
