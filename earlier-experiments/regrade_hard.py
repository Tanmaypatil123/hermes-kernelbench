"""
regrade_hard.py - re-run the hidden hard-tier grade on every saved solution with the CURRENT grader.

Why: the op recorder originally skipped a solution's first call, so a kernel replayed from a CUDA
graph looked like "no custom kernel" (Kimi K3's W4A16, 12 Oct). Every hard result is re-graded with
the fixed grader so all models are judged by the same rules. Writes <results>.regraded.jsonl next
to each results file and prints every verdict that changed; compare.py prefers the regraded files.

  python3 regrade_hard.py
"""
import glob, json
import hard_eval as he

files = sorted(set(glob.glob("hermes_results_*hard*.jsonl")) | set(glob.glob("hard_results_*.jsonl")))
for path in files:
    if path.endswith(".regraded.jsonl"):
        continue
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        if r.get("code"):
            res = he.grade(he.BY_ID[r["task"]], r["code"])
            res.pop("report", None)
            if res["status"] == "infra_error":   # Modal down or out of credit: stop spending
                raise SystemExit(f"infra_error while regrading {path} {r['task']}: {res.get('detail', '')[:300]}")
            old = (r["status"], r.get("peak_fraction"))
            r = {**r, **res, "status_before_regrade": r["status"]}
            new = (r["status"], r.get("peak_fraction"))
            flag = "CHANGED" if old[0] != new[0] else "same"
            pf = f"{new[1] * 100:.1f}%" if new[1] else "-"
            print(f"{flag:<8} {path:<45} {r['task']:<6} {old[0]} -> {new[0]} ({pf})", flush=True)
        out.append(r)
    with open(path.replace(".jsonl", ".regraded.jsonl"), "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in out))
