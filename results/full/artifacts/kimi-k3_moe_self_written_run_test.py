"""Drop-in replacement runner: same as test_kernel.py but survives modal's
303 long-poll redirect (re-GET the attempt-token URL until JSON arrives).
Also logs to .practice_log.jsonl exactly like test_kernel.py does.
"""
import ast, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
src_file = open(os.path.join(HERE, "test_kernel.py")).read()
tree = ast.parse(src_file)
CFG = None
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "CFG_JSON":
        CFG = json.loads(ast.literal_eval(node.value))

log_path = os.path.join(HERE, ".practice_log.jsonl")
used = sum(json.loads(l)["status"] != "infra_error" for l in open(log_path)) if os.path.exists(log_path) else 0
if used >= CFG["max_calls"]:
    print(f"BUDGET EXHAUSTED ({used}/{CFG['max_calls']}). Not sending.")
    sys.exit(1)

sol = open(os.path.join(HERE, "solution.py")).read()
program = CFG["program_template"].replace(CFG["placeholder"], json.dumps(sol), 1)
body = json.dumps({"code": program, "language": "triton", "gpu_type": CFG["gpu"], "timeout": CFG.get("timeout", 120)})

t0 = time.time()
body_path = "/tmp/_rt_body.json"
open(body_path, "w").write(body)
url = CFG["url"].rstrip("/") + "/execute"
hdrs = "/tmp/_rt_hdrs.txt"
resp = "/tmp/_rt_resp.txt"

def curl(u, extra, out, timeout):
    p = subprocess.run(["curl", "-sS", "-m", str(timeout), "-D", hdrs, "-o", out, "-w", "%{http_code}"] + extra + [u],
                       capture_output=True, text=True)
    return p.stdout.strip(), open(out, "rb").read()

status, raw = curl(url, ["-X", "POST", "-H", "Content-Type: application/json", "--data-binary", "@" + body_path], resp, 560)
attempt = 0
while status == "303" and attempt < 12:
    loc = None
    for line in open(hdrs):
        if line.lower().startswith("location:"):
            loc = line.split(":", 1)[1].strip()
    if not loc:
        break
    attempt += 1
    print(f"[303 -> poll {attempt}] {time.time()-t0:.0f}s elapsed", flush=True)
    status, raw = curl(loc, [], resp, 560)

try:
    rawj = json.loads(raw.decode())
    stdout, stderr = rawj.get("stdout") or "", rawj.get("stderr") or ""
except Exception as e:
    stdout, stderr = "", f"runner parse error: {e}; http={status}; head={raw[:300]!r}"

res = None
for line in stdout.splitlines():
    if line.startswith("RESULT_JSON:"):
        res = json.loads(line[len("RESULT_JSON:"):])
if res is None:
    res = {"status": "timeout" if "timed out" in stderr else "infra_error", "detail": stderr[-1500:]}

with open(log_path, "a") as f:
    f.write(json.dumps({"t": time.time(), "secs": round(time.time() - t0, 1), "status": res["status"],
                        "speedup": res.get("speedup"), "peak_fraction": res.get("peak_fraction"),
                        "detail": str(res.get("detail", ""))[:300]}) + "\n")
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
