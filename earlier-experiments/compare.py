"""
compare.py - Hermes with vs without the gpu-kernel-optimization skill, from the results files.

  python3 compare.py

Per model and tier it reports, for the baseline run (<model>_<tier>) and the skill run (<model>_<tier>_skill):
  pass            hidden-grader passes / attempts
  score           geomean speedup vs PyTorch (easy) or geomean % of peak bandwidth (hard), passes only
  regressed       passes whose submitted score is < 90% of the best score the agent saw in practice
  untested final  passes submitted right after a FAILED practice test (code changed without re-testing)
  practice        average practice runs used;   time  average agent wall time;   cost  model $ (Hermes estimate)
"""
import json, math, statistics
from pathlib import Path

MODELS = ["deepseek", "glm", "kimi"]


def load(run):
    p = Path(f"hermes_results_{run}.regraded.jsonl")   # hard tier: re-graded with the fixed grader
    if not p.exists():
        p = Path(f"hermes_results_{run}.jsonl")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else None


def practice(row):
    p = Path(row["folder"]) / ".practice_log.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def score(entry):
    return entry.get("peak_fraction") if entry.get("peak_fraction") is not None else entry.get("speedup")


def summarize(rows):
    ok = [r for r in rows if r["status"] == "correct"]
    scores = [score(r) for r in ok if score(r)]
    gm = math.exp(statistics.mean(math.log(s) for s in scores)) if scores else None
    regressed = untested = 0
    for r in ok:
        pr = practice(r)
        best = max((score(p) for p in pr if p["status"] == "correct" and score(p)), default=None)
        if best and score(r) and score(r) < 0.9 * best:
            regressed += 1
        if pr and pr[-1]["status"] != "correct":
            untested += 1
    cost = sum((r.get("usage") or {}).get("estimated_cost_usd") or 0 for r in rows)
    return {"pass": f"{len(ok)}/{len(rows)}", "gm": gm, "regressed": f"{regressed}/{len(ok)}",
            "untested": f"{untested}/{len(ok)}",
            "practice": statistics.mean(r["practice_calls"] for r in rows),
            "time": statistics.mean(r["agent_wall_s"] for r in rows), "cost": cost}


def fmt_score(gm, tier):
    if gm is None: return "-"
    return f"{gm:.2f}x" if tier == "easy" else f"{gm * 100:.1f}%"


def main():
    print(f"{'model':<9}{'tier':<6}{'arm':<9}{'pass':<7}{'score':<8}{'regressed':<11}{'untested':<10}"
          f"{'practice':<10}{'time':<8}{'cost':<7}")
    for tier in ("easy", "hard"):
        for m in MODELS:
            for arm, run in (("no skill", f"{m}_{tier}"), ("skill", f"{m}_{tier}_skill")):
                rows = load(run)
                if not rows:
                    print(f"{m:<9}{tier:<6}{arm:<9}(not run yet)"); continue
                s = summarize(rows)
                print(f"{m:<9}{tier:<6}{arm:<9}{s['pass']:<7}{fmt_score(s['gm'], tier):<8}{s['regressed']:<11}"
                      f"{s['untested']:<10}{s['practice']:<10.1f}{s['time']:<8.0f}${s['cost']:<6.3f}")
        print()


if __name__ == "__main__":
    main()
