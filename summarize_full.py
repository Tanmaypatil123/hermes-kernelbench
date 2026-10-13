"""
summarize_full.py - final table for the full KernelBench-Hard run with plain Hermes (no skill).

Reads results/full/. Which row counts for each (model, problem):
  * moe: results/full/hermes_results_<m>_moe_rerun.jsonl if present (re-run after the practice-tool
    fix); the broken-tool row stays in <m>_full.jsonl for reference.
  * a hidden grade that hit a GPU-service HTTP 500 (status infra_error) is replaced by
    results/full/regrade_<m>_<problem>.json: the same solution.py graded again.

  python3 summarize_full.py            # print the table
  python3 summarize_full.py --export   # also write solutions/<model>/<problem>/ and results/full/kernelbench_site.json
"""
import json, math, sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results" / "full"
SOL = Path(__file__).resolve().parent / "solutions"
# short name in file names, display name, slug for solutions/, model id, how it was served
MODELS = [("deepseek", "DeepSeek V4.1 Flash", "deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash", "OpenRouter"),
          ("qwen", "Qwen 3.8 Max", "qwen3.8-max", "qwen/qwen3.8-max-0902", "OpenRouter"),
          ("kimi", "Kimi K3", "kimi-k3", "moonshotai/kimi-k3", "OpenRouter"),
          ("gpt61", "GPT 6.1", "gpt-6.1", "gpt-6.1-sol", "Codex subscription"),
          ("glm53", "GLM 5.3", "glm-5.3", "z-ai/glm-5.3", "OpenRouter")]
TASKS = ["fp8", "kda", "paged", "topk", "moe", "w4a16"]
DIRS = {"fp8": "01_fp8_gemm", "kda": "02_kda_cutlass", "paged": "03_paged_attention",
        "topk": "05_topk_bitonic", "moe": "06_sonic_moe_swiglu", "w4a16": "07_w4a16_gemm"}

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []

def cost(r):
    c = (r.get("usage") or {}).get("estimated_cost_usd")
    return c if isinstance(c, (int, float)) else None

def rows_for(short):
    rows = {r["task"]: {**r, "_source": f"hermes_results_{short}_full.jsonl"} for r in load(R / f"hermes_results_{short}_full.jsonl")}
    for t in TASKS:
        rg = R / f"regrade_{short}_{t}.json"
        if t in rows and rows[t]["status"] == "infra_error" and rg.exists():
            rows[t] = {**rows[t], **json.load(open(rg)), "_note": "hidden grade hit a GPU-service HTTP 500; same solution.py re-graded"}
    for r in load(R / f"hermes_results_{short}_moe_rerun.jsonl"):
        rows["moe"] = {**r, "_source": f"hermes_results_{short}_moe_rerun.jsonl",
                       "_note": "re-run after the practice-tool fix (curl redirect bug broke long checks in the first run)"}
    return rows

def main():
    export = "--export" in sys.argv
    site, lines = [], []
    lines.append(f"{'model':<20}" + "".join(f"{t:>10}" for t in TASKS) + f"{'pass':>7}{'geomean*':>10}{'cost$':>8}{'time':>8}")
    detail = []
    for short, name, slug, mid, via in MODELS:
        rows = rows_for(short)
        if not rows: continue
        cells, pf, cs, wall, res = [], [], [], 0, {}
        for t in TASKS:
            r = rows.get(t)
            if not r: cells.append("–"); continue
            wall += r["agent_wall_s"]
            if cost(r) is not None: cs.append(cost(r))
            ok = r["status"] == "correct"
            if ok:
                pf.append(r["peak_fraction"])
                cells.append(f"{r['peak_fraction']*100:.1f}%" + ("T" if r["agent_exit"] == "timeout" else ""))
            else:
                cells.append("FAIL")
            practice = [p for p in r["practice_history"] if p != "infra_error"]
            res[t] = {"pass": ok, "peak": round(r["peak_fraction"] * 100, 1) if ok else None, "status": r["status"],
                      "timeout": r["agent_exit"] == "timeout", "minutes": round(r["agent_wall_s"] / 60, 1), "practice": practice}
            detail.append(f"  {name:<20}{t:<7}{r['status']:<16}exit={r['agent_exit']:<9}practice={practice}  torch_math={r.get('torch_math_ops')}")
            if export:
                d = SOL / slug / t
                d.mkdir(parents=True, exist_ok=True)
                if r.get("code"): (d / "solution.py").write_text(r["code"], encoding="utf-8")
                u = r.get("usage") or {}
                meta = {"model": mid, "served_via": via, "harness": "Hermes Agent, no skills", "problem": DIRS[t],
                        "gpu": r.get("gpu"), "status": r["status"], "peak_fraction": r.get("peak_fraction"),
                        "per_shape": r.get("per_shape"), "detail": r.get("detail"),
                        "torch_math_ops_at_runtime": r.get("torch_math_ops"), "triton_launches": r.get("triton_launches"),
                        "agent_exit": r["agent_exit"], "agent_minutes": round(r["agent_wall_s"] / 60, 1),
                        "practice_history": r["practice_history"],
                        "tokens": {k: u.get(k) for k in ("input_tokens", "output_tokens", "reasoning_tokens", "api_calls")},
                        "estimated_cost_usd": None if via.startswith("Codex") else cost(r),
                        "source": r["_source"], "note": r.get("_note"), "agent_final_message": (r.get("final_message") or "").strip()}
                (d / "result.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        g = math.exp(sum(math.log(x) for x in pf) / len(pf)) * 100 if pf else 0
        lines.append(f"{name:<20}" + "".join(f"{c:>10}" for c in cells)
                     + f"{len(pf):>5}/{len(rows)}{g:>9.1f}%{sum(cs):>8.2f}{wall/60:>7.0f}m")
        site.append({"name": name, "id": mid, "via": via, "slug": slug, "passed": len(pf), "geomean": round(g, 1),
                     "cost": None if via.startswith("Codex") else round(sum(cs), 2), "minutes": round(wall / 60), "results": res})
    lines += ["", "% = fraction of RTX PRO 6000 peak (KernelBench benchmark.py, geomean over shapes); T = agent hit the 45-min cap",
              "* geomean over the problems that model passed only; cost = Hermes' OpenRouter estimate (GPT 6.1: subscription)",
              "", "per-model details:"] + detail
    print("\n".join(lines))
    if export:
        (R / "full_kernelbench_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (R / "kernelbench_site.json").write_text(json.dumps(site, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {SOL}/, {R / 'full_kernelbench_summary.txt'}, {R / 'kernelbench_site.json'}")

if __name__ == "__main__":
    main()
