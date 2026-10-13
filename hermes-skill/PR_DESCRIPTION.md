## What does this PR do?

Adds `gpu-kernel-optimization`, an optional skill for writing and tuning custom GPU kernels (Triton, CUDA via `load_inline`), plus `kernel_ledger.py`, a small stdlib script that records every tested version of a kernel with its score and can restore and verify the best one.

The problem it solves came from benchmarking Hermes on kernel tasks (4 Triton tasks on a T4, 2 KernelBench-Hard problems on an RTX PRO 6000; hidden grader with fresh inputs, odd shapes and a runtime op recorder). Hermes plus open models reached correct kernels reliably, but lost results at the end of the session:

- **Regression:** the agent found a fast kernel, kept "optimizing", and submitted a slower one. DeepSeek V4.1 Flash softmax: 2.72x in practice, 1.05x to 1.10x submitted in 2 of 3 runs.
- **Untested final code:** the agent edited after its last test (or after a failed one) and submitted that.
- **No stopping rule:** sessions ran to the time limit without knowing how far from the hardware limit they were.

The skill gives a procedure (roofline first, one correct version, one change per test, deliberate stop, finish on the best version) and short patterns for fused elementwise ops, row reductions, top-k, weight-only quantized GEMM/GEMV and GEMM/attention. The ledger turns "finish on your best version" into two commands: `restore` and `verify`.

### Evidence: same harness, same models, with and without the skill (`--skills gpu-kernel-optimization`)

Easy tier: 4 Triton tasks x 3 runs per model, 3 practice tests per task, T4, speedup vs PyTorch eager.

| Model | Pass (no skill / skill) | Geomean speedup | Regressed submissions | Untested final code | Avg session | Model cost |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 12/12 / 12/12 | 1.72x -> **2.01x** | 2/12 -> **0/12** | 1/12 -> **0/12** | 108 s -> **56 s** | $0.175 -> **$0.097** |
| GLM 5.3 Flash | 12/12 / 12/12 | 1.95x -> 2.00x | 0/12 -> 0/12 | 1/12 -> **0/12** | 34 s -> 40 s | $0.044 -> $0.051 |
| Kimi K3 | 12/12 / 12/12 | 2.01x -> 2.00x | 0/12 -> 0/12 | 0/12 -> 0/12 | 47 s -> 50 s | $1.01 -> $1.05 |

Across 36 sessions per arm, regressed or untested submissions went from 4 to 0, and every model ends at the tier's ~2.0x ceiling. The weakest finisher (DeepSeek) gains most, and also halves its time and cost because the stopping rule ends unproductive tuning. Models that already finished well are unchanged.

Ledger use with the skill loaded: DeepSeek 12/12 sessions, Kimi 12/12, GLM 8/12.

Hard tier (2 KernelBench-Hard problems, 1 run per model per arm, % of 1.8 TB/s peak): all 12 sessions passed in both arms, but **speed differences are within measurement noise** (re-grading the same kernel moved results by up to 40%) and with n=1 this tier is anecdotal. The hard runs did surface real failure modes that v1.1.0 addresses:

- 3 of 3 models lost practice runs because KernelBench-style graders grep the source and flagged comments such as "does not use torch.topk". Added a pitfall.
- Models tried `tl.topk` / `tl.sort` signatures that did not compile on the test machine's Triton. Added a pitfall.
- With no test runs left after a failed test, models sometimes submitted an untested fix instead of restoring. Added an explicit out-of-budget rule, and moved the ledger command to the top of the skill (GLM used the ledger in only 8 of 12 easy sessions and 0 of 2 hard ones).

The A/B above was run with v1.0.0. Later versions add, untested in the benchmark so far:

- **v1.1.0:** the three pitfalls above and a `--timeout` for the ledger's test command.
- **v1.2.0:** cross-GPU guidance (`references/gpu-architectures.md`, sm75 to sm120, including that sm120 has no tcgen05/wgmma), a CUDA C++ checklist with a `load_inline` template (`references/cuda-kernel-checklist.md`), `scripts/gpu_info.py` (reads SM count, L2, shared-memory limit and arch on the target GPU), and two token savers motivated by the hard-tier cost: `run --quiet` (full output goes to a log file) and `restore` confirming the tested file in the same call. With the skill, hard-tier sessions made 190 API calls vs 99 without, largely from extra ledger turns re-sending long test output.

Not yet run on a GPU: the `load_inline` template in the CUDA checklist and `gpu_info.py`. I will run both on a T4 and an RTX PRO 6000 before marking this ready for review.

Benchmark harness, raw results and per-session logs: <link to the benchmark repo>.

## Related Issue

None. Not a duplicate of #62174 (a VS Code GPU monitoring surface); there is no kernel-writing skill today.

## Type of Change

- [x] 🎯 New skill (bundled or hub)

## Changes Made

- `optional-skills/mlops/gpu-kernel-optimization/SKILL.md`: the procedure, patterns, pitfalls and verification.
- `optional-skills/mlops/gpu-kernel-optimization/scripts/kernel_ledger.py`: `run` (with `--quiet`, `--timeout`) / `record` / `list` / `diff` / `restore` / `verify`; standard library only, cross-platform, state in `./.kernel_ledger/`.
- `optional-skills/mlops/gpu-kernel-optimization/scripts/gpu_info.py`: target-GPU facts for the roofline and tiling (needs PyTorch on the GPU machine).
- `optional-skills/mlops/gpu-kernel-optimization/references/gpu-architectures.md` and `references/cuda-kernel-checklist.md`.
- `tests/skills/test_gpu_kernel_ledger.py`: 8 tests (best-not-last restore, untested-edit rejection, passing-but-not-best warning, KernelBench-style output parsing, lower-is-better metrics, hung test command, quiet output with full log on disk, restore confirmation).
- `website/docs/...`: catalog row, sidebar entry and per-skill page, regenerated with `website/scripts/generate-skill-docs.py`.

## How to Test

1. `pytest tests/skills/test_gpu_kernel_ledger.py tests/skills/test_authoring_standards.py tests/skills/test_skill_docs_contract.py -q`
2. `hermes skills install gpu-kernel-optimization`, then in any folder with a `solution.py` and a test command: `hermes -z "Optimize solution.py; test with <cmd>" --skills gpu-kernel-optimization`, and check that `.kernel_ledger/` appears and `python3 <skill>/scripts/kernel_ledger.py verify` exits 0.
3. Without a GPU: the ledger tests use a fake test command, so step 1 runs anywhere.

## Checklist

### Code

- [x] I've read the Contributing Guide
- [x] My commit messages follow Conventional Commits
- [x] I searched for existing PRs to make sure this isn't a duplicate
- [x] My PR contains only changes related to this feature
- [ ] I've run `pytest tests/ -q` and all tests pass: ran `tests/skills/` (1770 passed; the same 3 tests fail on an unmodified checkout: `test_tool_error_strings_have_no_role_tag` and two `test_evolver_provider_routing` cases)
- [x] `python scripts/check` passes (12 checks)
- [x] I've added tests for my changes
- [x] I've tested on my platform: macOS (Apple Silicon) driving remote T4 and RTX PRO 6000 GPUs

### Documentation & Housekeeping

- [x] I've updated relevant documentation (generated skill docs)
- [x] `cli-config.yaml.example`: N/A
- [x] `CONTRIBUTING.md` / `AGENTS.md`: N/A

🤖 Generated with [Claude Code](https://claude.com/claude-code)
