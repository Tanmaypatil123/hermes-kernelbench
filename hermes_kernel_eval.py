"""
hermes_kernel_eval.py - benchmark HERMES AGENT on GPU-kernel tasks.

What changes vs kernel_eval.py (single-shot model):
  * Hermes gets a task folder and a practice tool (test_kernel.py) it can run from its terminal
    to test its kernel on YOUR GPU (via KernelIDE). It can iterate: write -> test -> fix.
  * The practice tool is deliberately limited: one input, one shape, no anti-cheat details,
    and a budget of GPU test calls (also caps your Modal spend).
  * After Hermes exits, the HIDDEN grader (same as kernel_eval.py: fresh inputs, odd shape,
    anti-cheat, timing, memory) scores solution.py. Hermes never sees that grader.

Extra things measured, because this is an agent:
  practice_calls     how many GPU test runs it used
  practice_status    what the practice tool said last  ->  "passed practice, failed hidden" = overfitting
  agent_wall_s       time the agent took;   agent_exit = finished / timeout / crashed
  usage              tokens + estimated cost from Hermes' --usage-file

Setup - Mac/Linux (Terminal), once:
  1. Install Hermes (the curl one-liner from the hermes-agent README), open a NEW terminal, then:
       hermes setup          (choose OpenRouter, paste your key)
  2. export KERNELIDE_URL="https://<your-workspace>--kernelide-executor-api.modal.run"
     export KERNELIDE_GPU="T4"
     export MODEL="<openrouter model id>"
  (Windows: PowerShell one-liner for Hermes, and $env:NAME = "..." instead of export.)

Run:
  python3 hermes_kernel_eval.py --tasks vector_add --runs 1     # smoke test first
  python3 hermes_kernel_eval.py --runs 3                        # full
Compare with the same model WITHOUT the agent:
  python3 kernel_eval.py --seeds 3
"""

import argparse, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

import kernel_eval as ke   # tasks, hidden grader, GPU runner, report
import hard_eval as he     # hard tier: KernelBench-Hard problems on RTX PRO 6000

# ------------------------------------------------------------------ the practice tool
# Written into every task folder. Standalone: only needs Python's standard library.
TEST_TOOL = r'''"""
test_kernel.py - practice test for your kernel on a real GPU.

Usage:   python3 test_kernel.py            (tests ./solution.py)
Checks correctness on ONE input against reference.py, and times it vs PyTorch.
You have a limited budget of runs; each run prints how many are left.
"""
import json, os, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.loads(CFG_JSON)
log_path = os.path.join(HERE, ".practice_log.jsonl")
# infra failures (network, certificates) are not the agent's fault, so they don't use up the budget
used = sum(json.loads(l)["status"] != "infra_error" for l in open(log_path, encoding="utf-8")) if os.path.exists(log_path) else 0
if used >= CFG["max_calls"]:
    print(f"BUDGET EXHAUSTED: you have used all {CFG['max_calls']} practice runs. Finish with your best solution.py.")
    sys.exit(1)
sol = os.path.join(HERE, "solution.py")
if not os.path.exists(sol):
    print("No solution.py found in this folder. Write it first."); sys.exit(1)
src = open(sol, encoding="utf-8").read()
program = CFG["program_template"].replace(CFG["placeholder"], json.dumps(src), 1)
t0 = time.time()
if CFG.get("local"):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try: exec(program, {"__name__": "__main__"})
        except SystemExit: pass
    stdout, stderr = buf.getvalue(), ""
else:
    body = json.dumps({"code": program, "language": "triton", "gpu_type": CFG["gpu"], "timeout": CFG.get("timeout", 120)}).encode()
    req = urllib.request.Request(CFG["url"].rstrip("/") + "/execute", data=body, headers={"Content-Type": "application/json"})
    try:
        try:
            with urllib.request.urlopen(req, timeout=CFG.get("http_timeout", 300)) as r:
                raw = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" not in str(e): raise
            # python.org builds on macOS ship without root certs; curl uses the system store.
            # -L: calls over ~150 s get a 303 to a "fetch result" URL (Modal web endpoints); follow it.
            # No "-X POST": with it curl re-POSTs to that URL and gets nothing; --data-binary already POSTs.
            import subprocess
            p = subprocess.run(["curl", "-s", "-L", "-m", str(CFG.get("http_timeout", 300)), "-H", "Content-Type: application/json",
                                "--data-binary", "@-", CFG["url"].rstrip("/") + "/execute"],
                               input=body, capture_output=True, check=True)
            raw = json.loads(p.stdout.decode("utf-8"))
        stdout, stderr = raw.get("stdout") or "", raw.get("stderr") or ""
    except Exception as e:
        stdout, stderr = "", f"GPU service error: {type(e).__name__}: {e}"
res = None
for line in stdout.splitlines():
    if line.startswith("RESULT_JSON:"):
        res = json.loads(line[len("RESULT_JSON:"):])
if res is None:
    res = {"status": "timeout" if "timed out" in stderr else "infra_error", "detail": stderr[-1500:]}
with open(log_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"t": time.time(), "secs": round(time.time() - t0, 1), "status": res["status"],
                        "speedup": res.get("speedup"), "peak_fraction": res.get("peak_fraction"), "detail": str(res.get("detail", ""))[:300]}) + "\n")
left = CFG["max_calls"] - used - (res["status"] != "infra_error")
print(f"status : {res['status']}")
if res.get("detail"): print(f"detail : {res['detail']}")
if "max_abs_err" in res: print(f"max abs error : {res['max_abs_err']:.3e}")
if "speedup" in res:
    print(f"your kernel   : {res['cand_ms']:.3f} ms")
    print(f"pytorch ref   : {res['ref_ms']:.3f} ms")
    print(f"speedup       : {res['speedup']:.2f}x  (higher is better)")
if res.get("report"): print(res["report"])
print(f"practice runs left: {left}")
'''

TASK_MD = """# Task: write a fast Triton kernel

{desc}

## Reference (your output must match this)
See `reference.py`:
```python
{reference}```

## What to deliver
Create `solution.py` in this folder that:
- imports torch, triton and triton.language as tl
- defines at least one `@triton.jit` kernel that does the computation
- defines `{signature}` which allocates the output (torch.empty_like etc. is fine), launches your kernel, and returns the result

## Rules
- The math must happen inside your Triton kernel. Do NOT compute the result with PyTorch ops
  (no torch.softmax, torch.exp, x.sum(), F.silu, a + b on tensors, @, etc.). Allocation is fine.
- Inputs are contiguous float32 CUDA tensors. Shapes can be odd sizes, so mask loads/stores.
- Your final solution will be graded on different inputs and shapes than the practice test.

## Testing
There is no GPU on this machine. Run `python3 test_kernel.py` to test `solution.py` on a remote GPU.
It checks one input and reports speed vs PyTorch. You have {max_calls} practice runs.

Goal: correct first, then as fast as you can. When done, reply with a one-line summary.
"""

PROMPT = ("Read TASK.md in the current directory and complete the task. "
          "Write solution.py, test it with `python3 test_kernel.py`, fix any problems, "
          "and stop when you are satisfied or out of practice runs.")

# ------------------------------------------------------------------ helpers
def kill_tree(p):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass

def make_task_dir(d, task, max_calls, local):
    d.mkdir(parents=True, exist_ok=True)
    (d / "reference.py").write_text("import torch\n\n" + task["reference"], encoding="utf-8")
    (d / "TASK.md").write_text(TASK_MD.format(max_calls=max_calls, **task), encoding="utf-8")
    placeholder = "\"__CANDIDATE_PLACEHOLDER__\""
    cfg = {"url": os.environ.get("KERNELIDE_URL", ""), "gpu": os.environ.get("KERNELIDE_GPU", "T4"),
           "max_calls": max_calls, "local": local, "placeholder": placeholder,
           "program_template": ke.build_program(task, "__CANDIDATE_PLACEHOLDER__", public=True)}
    (d / "test_kernel.py").write_text(f"CFG_JSON = {json.dumps(json.dumps(cfg))}\n" + TEST_TOOL, encoding="utf-8")

HARD_NOTE = """## How to test (this machine has no GPU)
Run `python3 test_kernel.py` in this folder. It sends your solution.py to a remote RTX PRO 6000,
runs check.py (every shape, 3 seeds, stress cases) and, if that passes, benchmark.py, and prints
their output. `status: correct` means check.py printed PASS. You have {max_calls} test runs.
reference.py, shapes.py and problem.yaml are in this folder. Your final solution.py is graded
again afterwards with additional checks. When done, reply with a one-line summary.
"""

def make_hard_task_dir(d, task, max_calls):
    d.mkdir(parents=True, exist_ok=True)
    src = he.KB / "problems" / task["dir"]
    for f in ("reference.py", "shapes.py", "problem.yaml"):
        (d / f).write_text((src / f).read_text(encoding="utf-8"), encoding="utf-8")
    (d / "TASK.md").write_text(he.task_prompt(task, HARD_NOTE.format(max_calls=max_calls)), encoding="utf-8")
    placeholder = "\"__CANDIDATE_PLACEHOLDER__\""
    cfg = {"url": os.environ.get("KERNELIDE_URL", ""), "gpu": he.GPU, "timeout": he.TIMEOUT,
           "http_timeout": he.TIMEOUT + 180, "max_calls": max_calls, "local": False, "placeholder": placeholder,
           "program_template": he.build_program(task, "__CANDIDATE_PLACEHOLDER__", public=True)}
    (d / "test_kernel.py").write_text(f"CFG_JSON = {json.dumps(json.dumps(cfg))}\n" + TEST_TOOL, encoding="utf-8")

def run_hermes(d, model, provider, timeout_s, hermes_cmd, skills=""):
    cmd = hermes_cmd + ["-z", PROMPT, "--usage-file", str(d / "usage.json")]
    if skills: cmd += ["--skills", skills]
    if model: cmd += ["-m", model]
    if provider: cmd += ["--provider", provider]
    kw = {"cwd": str(d), "stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True,
          "encoding": "utf-8", "errors": "replace"}
    if os.name == "nt": kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else: kw["start_new_session"] = True
    t0 = time.time()
    p = subprocess.Popen(cmd, **kw)
    try:
        out, err = p.communicate(timeout=timeout_s)
        exit_kind = "finished" if p.returncode == 0 else f"crashed(rc={p.returncode})"
    except subprocess.TimeoutExpired:
        kill_tree(p)
        out, err = p.communicate()
        exit_kind = "timeout"
    return exit_kind, time.time() - t0, out or "", err or ""

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="independent agent runs per task")
    ap.add_argument("--tasks", default="", help="comma list of task ids (default: all)")
    ap.add_argument("--max-practice", type=int, default=8, help="GPU practice runs the agent may use")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds before the agent is stopped")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--workdir", default=str(Path.home() / "kernel_runs"),
                    help="where task folders go (kept OUTSIDE your evals folder so the agent can't read the grader)")
    ap.add_argument("--run-name", default=time.strftime("hermes_%Y%m%d_%H%M%S"))
    ap.add_argument("--hermes", default="hermes", help="hermes command (path if not on PATH)")
    ap.add_argument("--local", action="store_true", help="dev only: grade on this CPU instead of the GPU")
    ap.add_argument("--hermes-skills", default="", help="skills to preload in every session, e.g. gpu-kernel-optimization")
    ap.add_argument("--tier", choices=["easy", "hard"], default="easy",
                    help="easy = our 4 Triton tasks on T4; hard = KernelBench-Hard problems on RTX PRO 6000")
    a = ap.parse_args()

    hermes_path = shutil.which(a.hermes) or (str(Path(a.hermes).resolve()) if Path(a.hermes).exists() else a.hermes)
    if not shutil.which(hermes_path) and not Path(hermes_path).exists():
        sys.exit(f"Can't find the hermes command ('{a.hermes}'). Install Hermes, open a NEW terminal window, "
                 f"and check that `hermes --version` works - or pass --hermes <full path>.")
    hermes_cmd = [sys.executable, hermes_path] if hermes_path.endswith(".py") else [hermes_path]
    model = os.environ.get("MODEL")
    all_tasks = he.TASKS if a.tier == "hard" else ke.TASKS
    by_id = {t["id"]: t for t in all_tasks}
    tasks = [by_id[x] for x in a.tasks.split(",")] if a.tasks else all_tasks
    root = Path(a.workdir) / a.run_name
    results_path = Path(f"hermes_results_{a.run_name}.jsonl")
    grade = ke.run_local if a.local else ke.run_on_gpu
    print(f"task folders : {root}\nresults file : {results_path}\nmodel        : {model}\nskills       : {a.hermes_skills or 'none'}\n")

    rows = []
    for run in range(a.runs):
        for t in tasks:
            d = root / f"{t['id']}_run{run}"
            if a.tier == "hard": make_hard_task_dir(d, t, a.max_practice)
            else: make_task_dir(d, t, a.max_practice, a.local)
            exit_kind, wall, out, err = run_hermes(d, model, a.provider, a.timeout, hermes_cmd, a.hermes_skills)

            # what the agent saw in practice
            log = d / ".practice_log.jsonl"
            practice = [json.loads(l) for l in open(log, encoding="utf-8")] if log.exists() else []
            usage = json.loads((d / "usage.json").read_text(encoding="utf-8")) if (d / "usage.json").exists() else None

            # hidden grade of the final solution
            sol = d / "solution.py"
            if sol.exists() and sol.read_text(encoding="utf-8").strip():
                if a.tier == "hard":
                    res = he.grade(t, sol.read_text(encoding="utf-8"))
                    res.pop("report", None)
                else:
                    res, _ = grade(ke.build_program(t, sol.read_text(encoding="utf-8")))
                code = sol.read_text(encoding="utf-8")
            else:
                res, code = {"status": "no_solution"}, None
            if exit_kind != "finished" and res["status"] == "no_solution":
                res["status"] = f"agent_{exit_kind}"

            row = {"seed": run, "task": t["id"], "skills": a.hermes_skills, **res, "agent_exit": exit_kind, "agent_wall_s": round(wall, 1),
                   "practice_calls": len(practice),
                   "practice_status": practice[-1]["status"] if practice else None,
                   "practice_history": [p["status"] for p in practice],
                   "usage": usage, "served_model": (usage or {}).get("model", model),
                   "final_message": out[-2000:], "agent_stderr": err[-2000:], "code": code, "folder": str(d)}
            rows.append(row)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            sp = f" {res['speedup']:.2f}x" if "speedup" in res else ""
            if "peak_fraction" in res: sp = f" {res['peak_fraction']*100:.1f}%peak"
            print(f"run={run} {t['id']:<11} hidden={res['status']}{sp:<8} practice={row['practice_history']} "
                  f"agent={exit_kind} {wall:.0f}s")
            if exit_kind.startswith("crashed"):
                print("   agent error (last lines):\n   " + "\n   ".join((err or out).strip().splitlines()[-6:]))

    he.report(rows) if a.tier == "hard" else ke.report(rows, a.runs)
    # agent-specific numbers
    n = len(rows)
    overfit = [r for r in rows if r["practice_status"] == "correct" and r["status"] != "correct"]
    print("--- agent behaviour ---")
    print(f"avg practice runs used                : {sum(r['practice_calls'] for r in rows) / n:.1f} of {a.max_practice}")
    print(f"never tested before finishing         : {sum(r['practice_calls'] == 0 for r in rows)}/{n}")
    print(f"passed practice but FAILED hidden     : {len(overfit)}/{n}  {[(r['task'], r['status']) for r in overfit]}")
    print(f"agent exits                           : { {k: sum(r['agent_exit'] == k for r in rows) for k in set(r['agent_exit'] for r in rows)} }")
    print(f"avg agent time                        : {sum(r['agent_wall_s'] for r in rows) / n:.0f}s")
    costs = [(r["usage"] or {}).get("estimated_cost_usd", (r["usage"] or {}).get("cost")) for r in rows]
    costs = [c for c in costs if isinstance(c, (int, float))]
    if costs: print(f"total model cost (Hermes estimate)    : ${sum(costs):.3f}")

if __name__ == "__main__":
    main()
