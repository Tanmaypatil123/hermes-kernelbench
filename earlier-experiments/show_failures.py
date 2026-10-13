"""
show_failures.py - print every failed attempt from results.jsonl so you can read it.

Run (in the same folder as results.jsonl):
  python3 show_failures.py            # all failures
  python3 show_failures.py matmul     # only one task
"""
import json, sys

only = sys.argv[1] if len(sys.argv) > 1 else None
with open("results.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

fails = [r for r in rows if not r["passed"] and (only is None or r["task"] == only)]
if not fails:
    print("No failures found" + (f" for task '{only}'" if only else "") + ".")
for r in fails:
    print("=" * 70)
    print(f"task={r['task']}  seed={r['seed']}  reason={r['reason']}")
    print("-" * 70)
    print(r["output"])
print("=" * 70)
print(f"{len(fails)} failure(s) shown.")
