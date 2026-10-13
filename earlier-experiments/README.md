# Earlier experiments (12 October 2026)

Kept for the record; superseded by the full run in `../results/full/`. These used the older
KernelIDE endpoint and an earlier version of the harness, so paths and some flags differ.

- **Easy tier** (`*_easy*`): 4 Triton tasks (vector_add, rmsnorm, softmax, swiglu) on a T4, `kernel_eval.py` (model
  alone) vs `hermes_kernel_eval.py` (Hermes), 3 runs each, for DeepSeek V4.1 Flash, GLM 5.3 Flash, Kimi K3.
- **Hard tier, 2 problems** (`*_hard*`): KernelBench-Hard topk + w4a16 only, model alone vs Hermes.
  `*.regraded.jsonl` = re-graded after the one-line `allclose` fix described in `../kb_hard/SOURCE.txt`.
- **Skill A/B** (`*_skill*`, `compare.py`): the same runs with the `gpu-kernel-optimization` skill
  (`../hermes-skill/`) preloaded.
- `toy_eval.py`: a first toy eval (10 small "write this Python function" problems) used to learn the basics.
