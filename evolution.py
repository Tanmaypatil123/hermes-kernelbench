"""
evolution.py - rebuild every practice run of the full KernelBench-Hard sweep from Hermes' own session log.

The result files only kept pass/fail per practice run. Hermes stores the whole conversation in
~/.hermes/state.db, so for each (model, problem) we replay the session:
  * the practice tool's own ledger (<run folder>/.practice_log.jsonl) -> status, % of peak, time
  * every write_file / patch on solution.py before it -> the exact solution.py that was tested,
    so we can diff attempt N against attempt N-1 and say what changed.
Sessions are found by their working folder (the run folder in the result row), so runs that timed
out (no session id in the result row) are found too. infra_error runs (GPU service failures) are kept
but marked; they don't count against the model.

  python3 evolution.py                 # print one line per run
  python3 evolution.py --diffs DIR     # also write DIR/<model>_<task>.md with each attempt's diff (for labelling)
  python3 evolution.py --export        # write results/full/evolution.json (merges evolution_labels.json)
"""
import difflib, json, math, os, re, sqlite3, sys
from pathlib import Path

from summarize_full import MODELS, TASKS, R, rows_for

DB = Path(os.path.expanduser("~/.hermes/state.db"))
LABELS = Path(__file__).resolve().parent / "evolution_labels.json"

def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []

def session_for(db, folder):
    # the moe re-run folders were used twice (broken-tool run, then the real one): take the latest
    row = db.execute("select id, started_at from sessions where cwd=? order by started_at desc limit 1", (folder,)).fetchone()
    return row

def replay(db, sid, ledger, k_ms):
    """One attempt per practice-ledger line (the tool's own record: status, % of peak, time).
    The session supplies the solution.py each attempt tested (edits replayed up to that moment)
    and the agent's notes in between. ms = k_ms / peak: % of peak is 1/time per shape, so the
    geomean time over shapes is a per-problem constant over the geomean peak fraction."""
    msgs = db.execute("select role, content, tool_calls, timestamp from messages where session_id=? order by id", (sid,)).fetchall()
    t0 = msgs[0][3] if msgs else 0
    src, events = "", []                            # (time, solution.py after this message, note)
    for role, content, tool_calls, ts in msgs:
        if role != "assistant": continue
        for c in json.loads(tool_calls or "[]"):
            f = c["function"]
            try: a = json.loads(f["arguments"])
            except json.JSONDecodeError: continue
            if not str(a.get("path", "")).endswith("solution.py"): continue
            if f["name"] == "write_file":
                src = a.get("content", "")
            elif f["name"] == "patch" and a.get("old_string") in src:
                src = src.replace(a["old_string"], a.get("new_string", ""), -1 if a.get("replace_all") else 1)
        events.append((ts, src, (content or "").strip()))
    attempts, i = [], 0
    for e in ledger:
        notes = []
        while i < len(events) and events[i][0] <= e["t"] - e.get("secs", 0):
            if events[i][2]: notes.append(events[i][2])
            i += 1
        pf = e.get("peak_fraction") if e["status"] == "correct" else None
        attempts.append({"status": e["status"], "peak": round(pf * 100, 2) if pf else None,
                         "geo_ms": round(k_ms / pf, 4) if pf and k_ms else None,
                         "minute": round((e["t"] - t0) / 60, 1), "src": events[i - 1][1] if i else "", "notes": notes})
    return attempts

def k_ms_by_task(runs_rows):
    """geomean(ms over shapes) * peak_fraction, from the hidden grades (constant per problem)."""
    ks = {}
    for r in runs_rows:
        if r["status"] == "correct" and r.get("per_shape"):
            ms = [s["ms"] for s in r["per_shape"]]
            ks.setdefault(r["task"], []).append(math.exp(sum(map(math.log, ms)) / len(ms)) * r["peak_fraction"])
    return {t: sum(v) / len(v) for t, v in ks.items()}

def collect():
    db = sqlite3.connect(DB)
    rows = {short: rows_for(short) for short, *_ in MODELS}
    k_ms = k_ms_by_task([r for rs in rows.values() for r in rs.values()])
    runs = []
    for short, name, slug, mid, via in MODELS:
        for task, r in rows[short].items():
            s = session_for(db, r["folder"])
            ledger = load_jsonl(Path(r["folder"]) / ".practice_log.jsonl")
            atts = replay(db, s[0], ledger, k_ms.get(task)) if s else []
            runs.append({"model": name, "slug": slug, "task": task, "session": s[0] if s else None,
                         "final": {"status": r["status"], "peak": round(r["peak_fraction"] * 100, 2) if r["status"] == "correct" else None,
                                   "ms": [x["ms"] for x in r.get("per_shape") or []], "timeout": r["agent_exit"] == "timeout"},
                         "recorded_history": r["practice_history"], "attempts": atts})
    return runs

def main():
    runs = collect()
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    for run in runs:
        got = [a["status"] for a in run["attempts"]]
        ok = "ok " if got == run["recorded_history"] else "MISMATCH"
        print(f"{ok} {run['slug']:<20}{run['task']:<7}" + " ".join(f"{a['status'][:4]}:{a['peak']}" if a["peak"] else a["status"][:4] for a in run["attempts"]))
        if ok != "ok ": print("     recorded:", run["recorded_history"])
    if "--diffs" in sys.argv:
        d = Path(sys.argv[sys.argv.index("--diffs") + 1]); d.mkdir(parents=True, exist_ok=True)
        for run in runs:
            lines, prev = [f"# {run['model']} / {run['task']}"], ""
            for i, a in enumerate(run["attempts"]):
                diff = "".join(difflib.unified_diff(prev.splitlines(True), a["src"].splitlines(True), n=1))
                lines += [f"\n## attempt {i + 1}: {a['status']} peak={a['peak']} geo_ms={a['geo_ms']} minute={a['minute']}",
                          "notes: " + " | ".join(n[:300] for n in a["notes"])[:900], "```diff", diff[:6000], "```"]
                prev = a["src"]
            (d / f"{run['slug']}_{run['task']}.md").write_text("\n".join(lines))
    if "--export" in sys.argv:
        out = []
        for run in runs:
            lab = labels.get(f"{run['slug']}/{run['task']}", {})
            atts = [{k: a[k] for k in ("status", "peak", "geo_ms", "minute")} | {"label": lab.get(str(i + 1))}
                    for i, a in enumerate(run["attempts"])]
            out.append({k: run[k] for k in ("model", "slug", "task", "final")} | {"attempts": atts, "story": lab.get("story")})
        (R / "evolution.json").write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", R / "evolution.json")

if __name__ == "__main__":
    main()
