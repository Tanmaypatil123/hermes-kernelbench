"""
show_kernel_failures.py - read every non-correct attempt in kernel_results_<model>.jsonl
(the file for the current $MODEL, or pass a .jsonl path, e.g. hermes_results_<run>.jsonl).

  python3 show_kernel_failures.py               # all failures
  python3 show_kernel_failures.py softmax       # one task
  python3 show_kernel_failures.py --all         # include correct ones too (to eyeball speedups/flags)
"""
import json, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
only = args[0] if args else None
show_all = "--all" in sys.argv
import os
from kernel_eval import results_file
path = next((a for a in sys.argv[1:] if a.endswith(".jsonl")), None) or results_file()   # default: current $MODEL
args = [a for a in args if not a.endswith(".jsonl")]
only = args[0] if args else None
print(f"reading {path}")
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
picked = [r for r in rows if (show_all or r["status"] != "correct") and (only is None or r["task"] == only)]

for r in picked:
    print("=" * 78)
    print(f"task={r['task']}  seed={r['seed']}  status={r['status']}  model={r.get('served_model')}")
    for k in ("detail", "failed_shape", "cheat_hits", "static_flags", "torch_math_ops",
              "triton_launches", "max_abs_err", "speedup", "ref_ms", "cand_ms", "ref_peak_mb", "cand_peak_mb"):
        if r.get(k) not in (None, [], ""):
            print(f"  {k:<16}: {r[k]}")
    print("-" * 78)
    print(r.get("code") or r.get("output") or "<empty>")
print("=" * 78)
print(f"{len(picked)} attempt(s) shown.")
