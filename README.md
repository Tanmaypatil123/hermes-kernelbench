# KernelBench-Hard × Hermes Agent

Five models, each driven by [Hermes Agent](https://github.com/NousResearch/hermes-agent) with no extra
skills, writing real GPU kernels for all six [KernelBench-Hard](https://github.com/Infatoshi/kernelbench.com)
problems on an NVIDIA RTX PRO 6000 (Blackwell, sm_120). Every kernel the agents submitted is in
[`solutions/`](solutions/), with its hidden-grader result next to it.

Write-up: <https://www.tanmaypatil.com/kernelbench>

## Results

% of the GPU's roofline peak reached by the final kernel (KernelBench's own `benchmark.py`, geomean over
shapes). Higher is better. ✗ = failed the hidden grade, † = the agent hit the 45-minute cap and its last
saved `solution.py` was graded.

| Model | fp8 | kda | paged | topk | moe | w4a16 | Solved | Geomean* | Model cost | Agent time |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 29.9% | 1.5% | **45.1%** | 2.2% | **10.5%** | 19.8% | 6/6 | **9.8%** | $0.87 | 160 min |
| Qwen 3.8 Max | **32.1%** | 1.2%† | 42.4% | **2.7%**† | 10.3%† | 17.7% | 6/6 | 9.6% | $3.58 | 244 min |
| Kimi K3 | 28.1% | **2.4%** | 25.8% | 2.0% | 9.0% | **23.7%** | 6/6 | 9.5% | $5.70 | 80 min |
| GPT 6.1 | 30.3% | 1.4% | 38.8% | 2.6% | 10.2% | 15.5% | 6/6 | 9.4% | subscription | 36 min |
| GLM 5.3 | ✗ | 1.5% | 39.4% | 1.5% | 9.1% | 15.7% | 5/6 | 6.6% | $2.83 | 74 min |

\* over the problems that model solved. Cost is Hermes' estimate of OpenRouter spend; GPT 6.1 ran on a
Codex subscription. One attempt per model per problem, so the top four (within 0.4 points) are a tie.

Model ids: `deepseek/deepseek-v4.1-flash`, `qwen/qwen3.8-max-0902`, `moonshotai/kimi-k3`, `z-ai/glm-5.3`
(all via OpenRouter) and `gpt-6.1-sol` (Hermes provider `openai-codex`).

## The problems

| id | KernelBench-Hard problem | What the agent writes |
|---|---|---|
| fp8 | `01_fp8_gemm` | fp8 e4m3 GEMM with per-channel scales |
| kda | `02_kda_cutlass` | Kimi Delta Attention chunk forward |
| paged | `03_paged_attention` | paged-KV attention decode with GQA |
| topk | `05_topk_bitonic` | row-wise fp32 top-k (no `torch.topk`/`sort`) |
| moe | `06_sonic_moe_swiglu` | MoE grouped GEMM + fused SwiGLU (no matmul ops) |
| w4a16 | `07_w4a16_gemm` | int4-weight × bf16 GEMM |

## How it works

1. `hermes_kernel_eval.py --tier hard` makes a task folder per problem (KernelBench's prompt,
   `reference.py`, shapes) and starts `hermes -z ...` in it. The machine has no GPU.
2. The agent tests with `test_kernel.py`, which sends `solution.py` to a remote RTX PRO 6000 through a
   KernelIDE executor (a small Modal web endpoint that runs code on a chosen GPU), runs KernelBench's `check.py` + `benchmark.py`, and
   prints the output. Budget: 6 test runs and 45 minutes per problem.
3. When the agent stops, `hard_eval.py` grades the final `solution.py` (hidden from the agent):
   KernelBench's correctness check (every shape, 3 seeds, numeric + property stress cases, banned-op grep)
   plus a runtime op recorder that fails any solution that actually runs a banned PyTorch op (matmul,
   softmax, topk, ...) even through `getattr` or a CUDA graph, and any solution with no custom kernel.
4. `python3 hard_eval.py --self-test` checks the grader itself on 12 known answers (good, wrong and
   cheating solutions) before trusting any score.

## Layout

```
solutions/<model>/<problem>/solution.py   the exact kernel that was graded
solutions/<model>/<problem>/result.json   score, per-shape timings, practice history, tokens, agent's last message
results/full/                             raw result rows (*.jsonl), logs, re-grades, summary table
results/full/artifacts/                   Kimi K3's self-written test runner (see Notes)
hermes_kernel_eval.py                     agent runner (task folders, practice tool, hidden grade)
hard_eval.py                              KernelBench-Hard grader + self-test
kernel_eval.py                            easy-tier tasks/grader (imported by the runner)
summarize_full.py                         builds the table; --export writes solutions/ and the site data
queue_full.sh, queue_moe_rerun.sh         how the runs were launched
kb_hard/                                  vendored KernelBench-Hard problems + eval code (MIT, Elliot Arledge)
hermes-skill/                             the gpu-kernel-optimization Hermes skill (NOT used in this run)
earlier-experiments/                      first easy-tier / 2-problem / skill A/B runs
```

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install openai pyyaml
# Hermes installed and `hermes setup` done; put your executor URL in env.local (git-ignored):
echo 'export KERNELIDE_URL="https://<your-workspace>--kernelide-executor-api.modal.run"' > env.local
source env.sh
python3 hard_eval.py --self-test
zsh queue_full.sh deepseek/deepseek-v4.1-flash deepseek            # one lane per model
zsh queue_full.sh gpt-6.1-sol gpt61 openai-codex                   # non-OpenRouter provider
# new runs write to the repo root; move them into results/full/ before summarize_full.py
python3 summarize_full.py --export
```

## Notes

- **Practice-tool bug (fixed mid-run).** Checks longer than ~150 s get a 303 redirect from the Modal web
  endpoint. The tool's `curl` fallback first didn't follow it, then re-POSTed to it (`-X POST`), so long
  moe checks came back empty. Fixed (`curl -L`, no `-X POST`); moe was re-run for the three models that hit
  it (`*_moe_rerun.jsonl`), and those re-runs are what the table shows.
- **Two re-grades.** Kimi K3's and Qwen 3.8 Max's moe hidden grades hit a GPU-service HTTP 500; the same
  saved `solution.py` was graded again (`regrade_*.json`).
- **Kimi K3 debugged the harness.** During the broken-tool moe run it worked out the redirect problem and
  wrote its own runner that follows it while still respecting and logging to the practice budget:
  `results/full/artifacts/kimi-k3_moe_self_written_run_test.py`.
- Torch ops listed in `torch_math_ops_at_runtime` are small helpers (routing metadata, autotune output
  comparisons), not the core math, which the recorder would reject.

## Credits

Problems and evaluation code: KernelBench-Hard by Elliot Arledge, MIT (`kb_hard/LICENSE`, source commit in
`kb_hard/SOURCE.txt`, one-line tolerance patch documented there). Agent: Hermes Agent by Nous Research.
