---
name: gpu-kernel-optimization
description: Write, test and tune fast, correct GPU kernels.
version: 1.2.0
author: Tanmay Patil (Tanmaypatil123), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GPU, CUDA, Triton, Kernels, Performance, Roofline, KernelBench, PyTorch]
    related_skills: [flash-attention, modal]
    requires_toolsets: [terminal]
---

# GPU Kernel Optimization

**Rule 1: run every test of your kernel through the ledger, and finish with `restore` + `verify`:**
`python3 ${HERMES_SKILL_DIR}/scripts/kernel_ledger.py run --quiet -- <test command>`.

A procedure for writing a custom GPU kernel (Triton, CUDA C++ via `load_inline`, CUTLASS)
that is correct first and fast second, and for finishing on your BEST version rather than your
last one. Measured on a kernel benchmark, agents most often lose results by optimizing a fast
kernel into a slower one and submitting it, or by editing after the final test.

## When to Use

- Replacing a PyTorch op or chain of ops with a custom kernel (fusion, softmax, norms, top-k,
  quantized GEMM/GEMV, attention).
- KernelBench-style tasks: a `reference.py`, a `solution.py` to write, and a test or benchmark command.
- Any loop of "edit kernel, run test, read numbers" where the test budget is limited.

## Prerequisites

- A way to run code on the target GPU: a local GPU, or the task's own test command (often remote).
- Python 3 for the ledger script (no extra packages). `gpu_info.py` needs PyTorch on the GPU machine.
- Reference files: `references/gpu-architectures.md` (per-GPU limits and features, sm75 to sm120)
  and `references/cuda-kernel-checklist.md` (CUDA C++ rules and a `load_inline` template).
  Open them with `skill_view` when the task targets a specific GPU or needs CUDA C++.

## How to Run

All commands go through the `terminal` tool, from the folder that holds `solution.py`.
Wrap EVERY test of the kernel in the ledger so each version and its score is kept:

```
python3 ${HERMES_SKILL_DIR}/scripts/kernel_ledger.py run --quiet --note "fused one-pass" -- <the task's test command>
```

With `--quiet` the ledger prints only the score lines of a pass, or the failure and the last lines of
a fail, and keeps the full output in `.kernel_ledger/logs/` (open it with `read_file` when you need
more). This keeps long test output out of the conversation. Then it prints one line: this version's
result and the best so far.
It parses `PASS` / `status: correct` / `FAIL` and a score (`peak_fraction`, `speedup`, `TFLOPS`, `gbps`).
Pass `--metric-regex 'time: ([0-9.]+) ms' --lower-is-better` when the score is a time.

## Quick Reference

| Command | Use |
|---|---|
| `kernel_ledger.py run -- <test cmd>` | Test the current file and record version + score |
| `kernel_ledger.py list` | All versions; marks the best; says if the file is untested |
| `kernel_ledger.py diff v2 v5` | What changed between two versions |
| `kernel_ledger.py restore` | Put the best passing version back and confirm it is the tested file |
| `gpu_info.py` | SM count, L2, shared-memory limit, arch, peak bandwidth of the GPU it runs on |
| `kernel_ledger.py verify` | Exit 0 only if the file is the best passing version |

## Procedure

1. **Read the contract.** `read_file` the task, `reference.py`, the shapes and dtypes, the tolerance,
   forbidden ops, and how many test runs you have. Keep one run in reserve for the end.
2. **Know the GPU, then compute the roofline.** Find the target GPU (the task text, the test
   output, or `python3 ${HERMES_SKILL_DIR}/scripts/gpu_info.py` where the kernel runs) and read its
   row in `references/gpu-architectures.md`: it decides which dtypes have tensor cores, how much
   shared memory a block may use, and how many blocks fill the GPU. Then for each shape: bytes =
   every input read once + every output written once; flops = the math. Ideal time = bytes / peak
   bandwidth (memory-bound) or flops / peak FLOP/s (compute-bound). Report results as a fraction of
   that ideal. Peak DRAM bandwidth (GB/s): T4 320, L4 300, A10G 600, L40S 864, A100-40GB 1555,
   A100-80GB 2039, RTX PRO 6000 1792, H100 SXM 3350 (PCIe 2000), H200 4800, B200 8000.
3. **Version 1 = the simplest correct kernel.** Mask every load and store, handle the odd shapes,
   compute in fp32. Test it with `kernel_ledger.py run`. Do not optimize before something passes.
4. **Optimize one idea per test.** Pick the change the roofline says matters most (usually: fewer
   passes over memory). Write a `--note` saying what changed. If the ledger says the new version is
   not better, either try a different idea or `restore`; do not stack more changes on a slower version.
5. **Stop on purpose.** Stop when you are above roughly 70% of the roofline on memory-bound shapes,
   when two ideas in a row gave under ~3% each, or when one test run is left.
6. **Finish on the best version.** Run `kernel_ledger.py restore` (it confirms the file is the
   tested best; run `verify` instead if you did not restore).
   Do not edit `solution.py` after the last passing test, not even comments: an untested file is an
   unknown result. If you changed it, test it again or restore.
   **Out of test runs and the last test failed?** Run `restore`. Never submit a fix you could not
   test, however small: the tested best version is a known result, the fix is a guess.

## Patterns

**Elementwise and fused elementwise** (add, activation, SwiGLU, bias+GELU). Memory-bound. One
program per contiguous block of 1024-4096 elements. The win comes from fusion: PyTorch writes and
re-reads every intermediate, a fused kernel reads inputs once and writes once. Expected speedup is
roughly (bytes PyTorch moves) / (bytes you move); a single op like `a + b` cannot beat ~1.0x.

**Row reductions** (softmax, RMSNorm, LayerNorm, log-softmax). If a row fits (up to ~8-16K fp32
values), use one program per row and keep the whole row in registers: one read, one write. Subtract
the row max before `exp`. Accumulate in fp32. Choose `num_warps` from the block size (about 4 up to
1024 elements, 8 up to 4096, 16 above). Longer rows: an online two-pass version (running max and
rescaled sum, then a write pass). This simple design is usually already near the roofline; extra
tricks (autotuning eviction hints, exp2 rewrites) rarely help and often hurt, so measure each one.

**Top-k / selection** (k much smaller than n). Never do k passes over the row (k x n reads). Use two
stages: (1) split each row into segments handled by many blocks, so all SMs work even when batch
is 1; each block keeps a small sorted top-k of its segment (per-thread buffers merged with warp
shuffles or a shared-memory bitonic network); (2) a merge kernel combines the segment candidates.
k = 1 is an argmax reduction. Pad non-power-of-two n with -inf.

**Weight-only quantized matmul** (int4/int8 weights, bf16/fp16 activations, W4A16). Dequantize
inside the matmul loop; never build the full-precision weight in memory. Decode (M = 1-16) is a GEMV
bound by reading the packed weights: spread work over many blocks along N (and split K if N alone
does not fill the GPU), load packed weights wide (one int32 = 8 int4 values), unpack with shifts and
masks, apply scale and zero-point per group, accumulate in fp32. Triton `tl.dot` needs M >= 16, so
pad M or use multiply-and-sum for M = 1. Prefill (M >= 64): tile with `tl.dot` and make the K tile
equal to the quantization group size so each scale loads once per tile.

**Plain GEMM and attention.** Library kernels (cuBLAS, cuDNN, FlashAttention) are hard to beat on
their own; wins come from fusing the neighbours (epilogue activation, dequant, masking). Use `tl.dot`
tiles and autotune BLOCK_M/N/K, `num_warps` and `num_stages`, then pin the winning config.

## Across GPUs

- **Tensor-core dtypes differ.** T4 (sm75) has no bf16 tensor cores; fp8 needs sm89+. Check the
  table before choosing a `tl.dot` or MMA dtype.
- **Shared memory differs 2-3x.** 64 KB per block on T4, 99 KB on sm86/sm89/sm120, 163 KB on A100,
  227 KB on H100/B200. A tile or `num_stages` setting tuned on one GPU may fail to launch on another.
- **Fill the GPU.** Blocks >= 2-4 x SM count (40 on T4, 188 on RTX PRO 6000). Batch-1 work needs a
  split along the long dimension; one block per row leaves most of a big GPU idle.
- **sm120 is not sm100.** RTX PRO 6000 / RTX 50xx have no `tcgen05`, TMEM or `wgmma`; H100 or B200
  tensor-core code will not build there. Compile for the exact arch (`gpu_info.py` prints the
  `-gencode` flag).
- **Prefer Triton for portability.** One Triton kernel runs from sm75 to sm120; reach for CUDA C++
  when you need warp shuffles, bit tricks or shared-memory layouts Triton cannot express.

## CUDA C++

Start from the `load_inline` template in `references/cuda-kernel-checklist.md` instead of writing
the build glue from scratch: it compiles for the current GPU, launches on PyTorch's stream and
checks launch errors. The short version of the checklist: coalesced, 128-bit loads with a scalar
tail; 64-bit indices; enough blocks for 2-4 waves; warp-shuffle reductions, then shared memory, then
a second kernel across blocks (no hot atomics); fp32 accumulation; `data_ptr<int64_t>()`, never
`data_ptr<long long>()`. Build errors arrive as import errors in the test output: read from the
first `error:` line.

## Pitfalls

- **Ending on the last version instead of the best.** The single most common way to lose a good
  result. The ledger exists for this: `restore` + `verify` before you stop.
- **Editing after the final test.** The submitted file must be byte-identical to a passing version.
- **Timing noise.** Time only after warm-up (the first call includes JIT compilation). On shared
  cloud GPUs, differences under ~5-10% are ties; re-run the top two versions before believing a
  small win. Autotuning picks noisy winners; pin configs once found.
- **Wrong answers that look right.** Missing masks pass on round sizes and break on odd ones;
  unmasked stores past the end can silently corrupt other memory. Use `other=-inf` for max-loads and
  `other=0` for sum-loads. Cast row offsets to int64 (`row.to(tl.int64) * n`) for large tensors.
- **`load_inline` builds.** Use `data_ptr<int64_t>()` / `data_ptr<float>()`, not
  `data_ptr<long long>()` (an undefined symbol on Linux). A compile error shows up as an import
  error in the test output, so read the whole message. In a long-lived Python process an extension
  with an unchanged `name` is not reloaded; change the name when iterating in one process.
- **Output contract.** Return exactly the dtype, shape and layout the reference returns (for
  example top-k: fp32 values plus int64 indices, sorted descending).
- **Forbidden names in comments.** Some graders search the source TEXT for forbidden ops, so a
  comment or docstring such as "does not use torch.topk" fails the check like a real call and wastes
  a test run. Never write a forbidden op's name anywhere in the file; describe the kernel instead.
- **Triton API drift.** `tl.sort`, `tl.topk`, `tl.reduce` and friends change signatures between
  Triton versions. Print `triton.__version__` and check `help(tl.sort)` with `execute_code` or the
  `terminal` tool before building a design on one; if the test machine differs from yours, prefer
  primitives you have seen compile there (`tl.max`, `tl.sum`, `tl.where`, `tl.load`/`tl.store`).
- **Do not cheat the grader.** Do not call the library op you were asked to replace (`torch.topk`,
  `torch.matmul`, `F.linear`, cuBLAS), not even through `getattr`, aliases or a different module; do
  not cache outputs between calls or special-case the test shapes or seeds. Graders record which
  ops actually run and test fresh inputs and shapes; such solutions score zero and waste the budget.

## Verification

- `python3 ${HERMES_SKILL_DIR}/scripts/kernel_ledger.py verify` exits 0: the file is the best
  passing version.
- `kernel_ledger.py list` shows the final version marked best, with its score as a fraction of the
  roofline you computed in step 2.
