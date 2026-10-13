"""
hard_eval.py - the HARD tier: all six KernelBench-Hard problems on an RTX PRO 6000, via KernelIDE.

Problems (from github.com/Infatoshi/kernelbench.com, benchmarks/hard/problems-rtxpro6000, MIT,
(c) Elliot Arledge; vendored in kb_hard/, commit in kb_hard/SOURCE.txt):
  fp8    01_fp8_gemm          fp8 e4m3 GEMM, 4 shapes, compute-bound
  kda    02_kda_cutlass       Kimi Delta Attention chunk forward, 4 shapes, compute-bound
  paged  03_paged_attention   paged attention decode (GQA), 5 shapes, memory-bound
  topk   05_topk_bitonic      fp32 top-k, 5 shapes, memory-bound
  moe    06_sonic_moe_swiglu  MoE grouped GEMM + SwiGLU, 3 shapes, compute-bound
  w4a16  07_w4a16_gemm        int4-weight x bf16 GEMM, 5 shapes, memory-bound at decode

Grading = KernelBench's own check.py (every shape, 3 seeds, stress + property cases, forbidden-op grep)
          + benchmark.py (score = fraction of the GPU's 1800 GB/s peak, geomean over shapes)
          + our runtime op recorder (hidden only): records every PyTorch op the solution runs, so
            a forbidden op reached without its name in the text (getattr tricks, unfused matmul)
            is caught, and a "solution" with no custom kernel at all is rejected.

Run:
  python3 hard_eval.py --self-test                    # grader check on known answers (do this first)
  python3 hard_eval.py --alone --seeds 1              # model alone, one shot per problem
Hermes on this tier: python3 hermes_kernel_eval.py --tier hard ...
"""

import argparse, base64, io, json, math, os, re, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB = HERE / "kb_hard"
GPU = os.environ.get("KERNELIDE_HARD_GPU", "RTX-PRO-6000")
TIMEOUT = 600

# "runtime_forbidden" = aten ops that, seen at runtime, mean PyTorch did the core math (not the custom kernel).
# Stricter than KernelBench's text grep, which only sees names written in solution.py.
_MATMUL = ["mm", "matmul", "addmm", "bmm", "baddbmm", "linear", "einsum"]
TASKS = [
    {"id": "fp8", "dir": "01_fp8_gemm", "runtime_forbidden": _MATMUL + ["_scaled_mm", "_scaled_mm_v2"]},
    {"id": "kda", "dir": "02_kda_cutlass", "runtime_forbidden": _MATMUL},
    {"id": "paged", "dir": "03_paged_attention",
     "runtime_forbidden": _MATMUL + ["_softmax", "softmax", "scaled_dot_product_attention",
                                     "_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
                                     "_scaled_dot_product_cudnn_attention", "_flash_attention_forward",
                                     "_efficient_attention_forward"]},
    {"id": "topk", "dir": "05_topk_bitonic",
     # aten ops that mean "PyTorch did the selection for you"
     "runtime_forbidden": ["topk", "sort", "argsort", "kthvalue", "msort"]},
    {"id": "w4a16", "dir": "07_w4a16_gemm",
     # a matmul op at runtime means dequant-then-matmul (not fused) or F.linear in disguise
     "runtime_forbidden": ["linear", "mm", "matmul", "addmm", "bmm", "baddbmm", "_weight_int4pack_mm"]},
    {"id": "moe", "dir": "06_sonic_moe_swiglu", "runtime_forbidden": _MATMUL + ["_grouped_mm"]},
]
TASKS.sort(key=lambda t: t["dir"])
BY_ID = {t["id"]: t for t in TASKS}

# ------------------------------------------------------------------ the program that runs on the GPU
REMOTE = r'''
import base64, io, json, os, re, subprocess, sys, tarfile, time
out = {"stage": "setup", "problem": PROBLEM, "public_mode": PUBLIC}
def emit():
    print("RESULT_JSON:" + json.dumps(out)); sys.stdout.flush()

deps = "/tmp/kbdeps"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", deps, "pyyaml", "hypothesis", "ninja", "einops"],
                   capture_output=True, text=True)
if r.returncode:
    out.update(status="infra_error", detail="pip install failed: " + r.stderr[-600:]); emit(); sys.exit(0)
root = "/tmp/kb"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE_B64))).extractall(root)
pdir = os.path.join(root, "problems", PROBLEM)
open(os.path.join(pdir, "solution.py"), "w").write(SOLUTION_SRC)
env = dict(os.environ, PYTHONPATH=deps + os.pathsep + root, PATH=deps + "/bin" + os.pathsep + os.environ.get("PATH", ""),
           TORCH_EXTENSIONS_DIR="/tmp/torch_ext", PYTHONDONTWRITEBYTECODE="1")
import torch
out["gpu"] = torch.cuda.get_device_name(0)

def run(script, timeout):
    try:
        p = subprocess.run([sys.executable] + script, cwd=pdir, env=env, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        return "timeout", ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))

# ---- 1. KernelBench's own correctness check
out["stage"] = "check"
t0 = time.time()
rc, log = run(["check.py"], 360)
out["check_s"] = round(time.time() - t0, 1)
tail = log.strip()[-2500:]
out["check_tail"] = tail
lines = log.strip().splitlines()
if rc == "timeout":
    out.update(status="timeout", detail="check.py took over 360 s")
elif lines and lines[-1].strip() == "PASS":
    out["check"] = "PASS"
elif "forbidden op used" in log:
    out.update(status="forbidden_op", detail=next(l for l in lines if "forbidden op used" in l))
elif "solution import error" in log or "state_dict mismatch" in log:
    out.update(status="import_error", detail=next((l for l in lines if l.startswith("FAIL")), tail[-400:]))
elif any(l.startswith("FAIL") for l in lines):
    out.update(status="wrong_output", detail=next(l for l in lines if l.startswith("FAIL"))[:600])
else:
    out.update(status="runtime_error", detail=tail[-800:])

# ---- 2. runtime op recorder (hidden grade only)
if out.get("check") == "PASS" and not PUBLIC:
    out["stage"] = "op_recorder"
    rec = r"""
import json, sys
sys.path.insert(0, ".")
import torch
from torch.utils._python_dispatch import TorchDispatchMode
LAUNCHES = [0]
try:
    from triton.runtime.jit import JITFunction
    _orig = JITFunction.run
    def _count(self, *a, **k):
        LAUNCHES[0] += 1
        return _orig(self, *a, **k)
    JITFunction.run = _count
except Exception:
    pass
class OpLog(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.ops = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))
import reference, shapes, solution
dev = torch.device("cuda:0")
res = {}
try:
    from check import _apply_shape          # problems whose shape keys differ from reference's globals
except ImportError:
    def _apply_shape(m, shape):
        for k, v in shape.items(): setattr(m, k, v)
for i, shape in enumerate(shapes.SHAPES[:2]):
    _apply_shape(reference, shape)
    init = reference.get_init_inputs()
    ref = reference.Model(*init).to(dev).eval()
    sol = solution.Model(*init).to(dev).eval()
    sol.load_state_dict(ref.state_dict(), strict=True)
    torch.manual_seed(7)
    inp = [t.to(dev) for t in reference.get_inputs()]
    with torch.no_grad():
        # Record the first call too: compilation, autotuning and CUDA-graph capture happen there, and
        # a captured graph replays its kernels on later calls without going through Python at all.
        log = OpLog()
        with log:
            sol(*inp); torch.cuda.synchronize()
            sol(*inp); torch.cuda.synchronize()
    res.setdefault("ops", []).extend(log.ops)
    res["launches"] = res.get("launches", 0) + LAUNCHES[0]
print("REC_JSON:" + json.dumps(res))
"""
    open(os.path.join(pdir, "_recorder.py"), "w").write(rec)
    rc, log = run(["_recorder.py"], 240)
    rec_res = None
    for l in log.splitlines():
        if l.startswith("REC_JSON:"): rec_res = json.loads(l[9:])
    if rec_res is None:
        out.update(status="harness_error", detail="op recorder failed: " + log[-800:]); emit(); sys.exit(0)
    ALLOWED = {"empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like",
               "new_empty", "new_empty_strided", "new_zeros", "new_ones", "new_full", "fill_", "zero_", "view",
               "_unsafe_view", "reshape", "_reshape_alias", "as_strided", "t", "transpose", "permute", "expand",
               "contiguous", "clone", "copy_", "detach", "alias", "select", "slice", "unsqueeze", "squeeze",
               "lift_fresh", "lift_fresh_copy", "_to_copy", "to", "flatten", "view_as", "arange", "sym_size",
               "sym_stride", "sym_numel", "is_contiguous", "unbind", "split", "narrow", "item", "_local_scalar_dense"}
    ops = sorted(set(rec_res["ops"]))
    math_ops = [o for o in ops if o not in ALLOWED]
    out["torch_math_ops"] = math_ops
    out["triton_launches"] = rec_res["launches"]
    hits = [o for o in ops if o in RUNTIME_FORBIDDEN]
    custom = rec_res["launches"] > 0 or re.search(r"load_inline|cpp_extension|cuda\.core|cupy|ctypes", SOLUTION_SRC)
    if hits:
        out.update(status="cheat_suspected", detail="forbidden op ran at runtime: " + ", ".join(hits))
    elif not custom:
        out.update(status="no_custom_kernel", detail="no Triton launch and no compiled extension; torch ops: " + ", ".join(math_ops))

# ---- 3. KernelBench's benchmark (roofline fraction)
if out.get("check") == "PASS" and "status" not in out:
    out["stage"] = "benchmark"
    rc, log = run(["benchmark.py"], 300)
    shapes_ = [dict(re.findall(r"(\w+)=([\d.]+)", l)) for l in log.splitlines()
               if l.startswith("shape=") and "variant=solution" in l and "gbps=" in l]
    m = re.search(r"peak_fraction: ([\d.]+)", log)
    if rc == "timeout" or not m:
        out.update(status="runtime_error" if rc != "timeout" else "timeout", detail="benchmark.py: " + log[-800:])
    else:
        out.update(status="correct", peak_fraction=float(m.group(1)),
                   per_shape=[{"gbps": float(s["gbps"]), "ms": float(s["ms"])} for s in shapes_],
                   bench_tail=log.strip()[-1500:])

out["stage"] = "done"
rep = [f"status: {out.get('status')}"]
if out.get("detail"): rep.append(f"detail: {out['detail']}")
rep.append("--- check.py output (tail) ---\n" + out.get("check_tail", "")[-1500:])
if out.get("bench_tail"): rep.append("--- benchmark.py output ---\n" + out["bench_tail"])
out["report"] = "\n".join(rep)
emit()
'''

_BUNDLE = None
def bundle_b64():
    """kb_hard/ (src + problems, no solutions) as a base64 tar.gz."""
    global _BUNDLE
    if _BUNDLE is None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for p in sorted(KB.rglob("*.py")) + sorted(KB.rglob("*.yaml")):
                if "__pycache__" not in p.parts:
                    tf.add(p, arcname=str(p.relative_to(KB)))
        _BUNDLE = base64.b64encode(buf.getvalue()).decode()
    return _BUNDLE

def build_program(task, solution_src, public=False):
    return (f"BUNDLE_B64 = {json.dumps(bundle_b64())}\n"
            f"PROBLEM = {json.dumps(task['dir'])}\n"
            f"SOLUTION_SRC = {json.dumps(solution_src)}\n"
            f"RUNTIME_FORBIDDEN = {json.dumps(task['runtime_forbidden'])}\n"
            f"PUBLIC = {public}\n" + REMOTE)

def run_on_gpu(program, timeout=TIMEOUT, gpu=None):
    url = os.environ["KERNELIDE_URL"].rstrip("/") + "/execute"
    body = json.dumps({"code": program, "language": "triton", "gpu_type": gpu or GPU, "timeout": timeout}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout + 180) as r:
            raw = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"status": "infra_error", "detail": f"{type(e).__name__}: {e}"}, {"wall_s": time.time() - t0}
    raw["wall_s"] = time.time() - t0
    for line in (raw.get("stdout") or "").splitlines():
        if line.startswith("RESULT_JSON:"):
            res = json.loads(line[len("RESULT_JSON:"):])
            res["gpu_wall_s"] = raw.get("execution_time")
            return res, raw
    err = raw.get("stderr") or ""
    st = "timeout" if "timed out" in err.lower() else "infra_error"
    return {"status": st, "detail": ("no RESULT_JSON; stderr: " + err)[-800:]}, raw

def grade(task, solution_src, public=False):
    res, _ = run_on_gpu(build_program(task, solution_src, public))
    return res

# ------------------------------------------------------------------ the agent-facing task text
def task_prompt(task, practice_note):
    """KernelBench's own PROMPT.txt, with the GPU-access line adapted to our setup."""
    p = (KB / "problems" / task["dir"] / "PROMPT.txt").read_text()
    p = p.replace("verify correctness by running `python check.py` and reading the output",
                  "verify correctness with the test tool described below")
    p = p.replace("If `python check.py` hasn't printed PASS, you're not done.",
                  "If the test tool hasn't shown `status: correct`, you're not done.")
    return p + "\n\n" + practice_note

# ------------------------------------------------------------------ self-test answers
TOPK_GOOD = r'''
import torch, triton, triton.language as tl

@triton.jit
def _topk_rows(x_ptr, v_ptr, i_ptr, n, k, BLOCK: tl.constexpr):
    # k passes per row; pass j finds the largest (value, -index) strictly below the previous pick
    row = tl.program_id(0)
    base = x_ptr + row.to(tl.int64) * n
    pv = float("inf"); pi = -1
    for j in range(k):
        best_v = -float("inf"); best_i = n
        for s in range(0, n, BLOCK):
            idx = s + tl.arange(0, BLOCK)
            x = tl.load(base + idx, mask=idx < n, other=-float("inf"))
            ok = (idx < n) & ((x < pv) | ((x == pv) & (idx > pi)))
            xv = tl.where(ok, x, -float("inf"))
            cv = tl.max(xv, axis=0)
            ci = tl.min(tl.where(ok & (xv == cv), idx, n), axis=0)
            better = (cv > best_v) | ((cv == best_v) & (ci < best_i))
            best_i = tl.where(better, ci, best_i)
            best_v = tl.where(better, cv, best_v)
        tl.store(v_ptr + row * k + j, best_v)
        tl.store(i_ptr + row * k + j, best_i.to(tl.int64))
        pv = best_v; pi = best_i

class Model(torch.nn.Module):
    def __init__(self, batch, n, k):
        super().__init__()
        self.batch, self.n, self.k = batch, n, k
        self.register_buffer("_dummy", torch.zeros(1))
    def forward(self, x):
        x = x.contiguous()
        b, n = x.shape
        v = torch.empty((b, self.k), dtype=torch.float32, device=x.device)
        i = torch.empty((b, self.k), dtype=torch.int64, device=x.device)
        _topk_rows[(b,)](x, v, i, n, self.k, BLOCK=1024)
        return v, i
'''
# same interface, but asks PyTorch for top-k without writing its name (their grep can't see it)
TOPK_SNEAKY = TOPK_GOOD.split("class Model")[0] + r'''
class Model(torch.nn.Module):
    def __init__(self, batch, n, k):
        super().__init__()
        self.k = k
        self.register_buffer("_dummy", torch.zeros(1))
    def forward(self, x):
        _topk_rows[(1,)](x, torch.empty(1, device=x.device), torch.empty(1, dtype=torch.int64, device=x.device), 1, 0, BLOCK=16)
        return getattr(torch, "to" + "pk")(x, self.k, dim=-1)
'''
TOPK_WRONG = TOPK_GOOD.replace("(x < pv) | ((x == pv) & (idx > pi))", "(x < pv)")   # drops tied values... and
TOPK_WRONG = TOPK_WRONG.replace("tl.store(v_ptr + row * k + j, best_v)", "tl.store(v_ptr + row * k + j, best_v * 0.999)")  # perturbs values

W4A16_GOOD = r'''
import torch, triton, triton.language as tl

@triton.jit
def _w4a16(x_ptr, w_ptr, s_ptr, z_ptr, o_ptr, M, N, K,
           BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk2 = tl.arange(0, G // 2)                       # packed rows in one group
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, K // G):
        kb = g * G
        xe = tl.load(x_ptr + rm[:, None] * K + (kb + 2 * rk2)[None, :], mask=rm[:, None] < M, other=0.0)
        xo = tl.load(x_ptr + rm[:, None] * K + (kb + 2 * rk2 + 1)[None, :], mask=rm[:, None] < M, other=0.0)
        p = tl.load(w_ptr + (kb // 2 + rk2)[:, None] * N + rn[None, :], mask=rn[None, :] < N, other=0)
        s = tl.load(s_ptr + g * N + rn, mask=rn < N, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + g * N + rn, mask=rn < N, other=0.0).to(tl.float32)
        lo = (((p & 0xF).to(tl.float32) - z[None, :]) * s[None, :]).to(tl.bfloat16)
        hi = (((p >> 4).to(tl.float32) - z[None, :]) * s[None, :]).to(tl.bfloat16)
        acc += tl.dot(xe, lo) + tl.dot(xo, hi)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))

class Model(torch.nn.Module):
    def __init__(self, M, N, K, group_size=128):
        super().__init__()
        self.M, self.N, self.K, self.group_size = M, N, K, group_size
        self.register_buffer("w_q", torch.zeros(K // 2, N, dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros(K // group_size, N, dtype=torch.bfloat16))
        self.register_buffer("zeros", torch.zeros(K // group_size, N, dtype=torch.bfloat16))
    def forward(self, x):
        x = x.to(torch.bfloat16).contiguous()
        M, K = x.shape
        o = torch.empty((M, self.N), dtype=torch.bfloat16, device=x.device)
        BM = 16 if M <= 16 else 64
        grid = (triton.cdiv(M, BM), triton.cdiv(self.N, 64))
        _w4a16[grid](x, self.w_q, self.scales, self.zeros, o, M, self.N, K, BM=BM, BN=64, G=self.group_size)
        return o
'''
W4A16_UNFUSED = (KB / "problems" / "07_w4a16_gemm" / "reference.py").read_text() if KB.exists() else ""
W4A16_WRONG = W4A16_GOOD.replace("acc += tl.dot(xe, lo) + tl.dot(xo, hi)", "acc += tl.dot(xe, hi) + tl.dot(xo, lo)")  # swapped nibbles

# The honest fused kernel, captured into a CUDA graph on the first call and replayed afterwards.
W4A16_GRAPH = W4A16_GOOD.replace("class Model(torch.nn.Module):", "class _Eager(torch.nn.Module):") + r"""
class Model(_Eager):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self._g = None
    def forward(self, x):
        if self._g is None:
            self._x = x.to(torch.bfloat16).contiguous().clone()
            _Eager.forward(self, self._x); torch.cuda.synchronize()
            self._g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._g):
                self._o = _Eager.forward(self, self._x)
        self._x.copy_(x); self._g.replay()
        return self._o.clone()
"""
# torch.topk hidden inside a CUDA graph: invisible on replay, visible only while it is captured.
TOPK_GRAPH_CHEAT = TOPK_GOOD.split("class Model")[0] + r"""
class Model(torch.nn.Module):
    def __init__(self, batch, n, k):
        super().__init__(); self.k = k; self._g = None
        self.register_buffer("_dummy", torch.zeros(1))
    def forward(self, x):
        if self._g is None or self._x.shape != x.shape:
            self._x = x.clone()
            getattr(torch, "to" + "pk")(self._x, self.k, dim=-1); torch.cuda.synchronize()
            self._g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._g):
                self._v, self._i = getattr(torch, "to" + "pk")(self._x, self.k, dim=-1)
        self._x.copy_(x); self._g.replay()
        return self._v.clone(), self._i.clone()
"""

SELF_TESTS = [("topk good (slow, honest Triton)", "topk", TOPK_GOOD, "correct"),
              ("topk sneaky getattr(torch,'to'+'pk')", "topk", TOPK_SNEAKY, "cheat_suspected"),
              ("topk wrong values", "topk", TOPK_WRONG, "wrong_output"),
              ("w4a16 good (fused Triton)", "w4a16", W4A16_GOOD, "correct"),
              ("w4a16 unfused reference (dequant then @)", "w4a16", W4A16_UNFUSED, "cheat_suspected"),
              ("w4a16 swapped nibbles", "w4a16", W4A16_WRONG, "wrong_output"),
              ("w4a16 honest kernel in a CUDA graph", "w4a16", W4A16_GRAPH, "correct"),
              ("topk torch.topk hidden in a CUDA graph", "topk", TOPK_GRAPH_CHEAT, "cheat_suspected")]
# The other four problems: their own PyTorch reference handed in as the "solution". It must pass
# KernelBench's check.py (proves the problem runs end to end on this GPU) and then be caught by the
# op recorder, because the math ran in PyTorch, not a custom kernel.
def _bare_reference(tid):
    """reference.py without docstrings/comments (they name the forbidden ops, which trips the text grep)."""
    import ast
    tree = ast.parse((KB / "problems" / BY_ID[tid]["dir"] / "reference.py").read_text())
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree).replace("_naive_chunk_kda", "_ref_kda")
for _tid in ("fp8", "kda", "paged", "moe"):
    _ref = _bare_reference(_tid) if KB.exists() else ""
    SELF_TESTS.append((f"{_tid} reference copied as solution", _tid, _ref, "cheat_suspected"))

# ------------------------------------------------------------------ model alone (one shot)
def ask_model(task, seed, temperature):
    from openai import OpenAI
    ref = (KB / "problems" / task["dir"] / "reference.py").read_text()
    prompt = ((KB / "problems" / task["dir"] / "PROMPT.txt").read_text()
              + f"\n\nreference.py:\n```python\n{ref}```\n\n"
              "You get ONE attempt and cannot run anything. Reply with only the complete solution.py "
              "in a single ```python code block.")
    r = OpenAI().chat.completions.create(model=os.environ["MODEL"], seed=seed, temperature=temperature,
                                         max_tokens=32000, messages=[{"role": "user", "content": prompt}])
    c = r.choices[0]
    return (c.message.content or ""), c.finish_reason, getattr(r, "model", None), getattr(r, "usage", None)

def extract_code(text):
    blocks = re.findall(r"```[ \t]*(?:python|py|Python)?[ \t]*\r?\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else None

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def report(rows):
    n = len(rows)
    ok = [r for r in rows if r["status"] == "correct"]
    lo, hi = wilson(len(ok), n)
    print("\n=== HARD TIER RESULTS ===")
    print(f"model requested : {os.environ.get('MODEL')}")
    print(f"model(s) served : {sorted(set(str(r.get('served_model')) for r in rows))}")
    print(f"gpu             : {sorted(set(str(r.get('gpu')) for r in rows))}")
    print(f"pass            : {len(ok)}/{n}   95% CI [{lo:.2f}, {hi:.2f}]")
    for t in TASKS:
        tr = [r for r in rows if r["task"] == t["id"]]
        if not tr: continue
        fr = [r.get("peak_fraction") for r in tr if r["status"] == "correct"]
        print(f"   {t['id']:<6} " + "  ".join(
            f"{r['status']}" + (f"({r['peak_fraction']*100:.1f}% of peak)" if r['status'] == 'correct' else "") for r in tr))
    statuses = {}
    for r in rows: statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"outcomes        : {statuses}")
    gs = [r.get("gpu_wall_s") or 0 for r in rows]
    print(f"GPU seconds (hidden grading only): {sum(gs):.0f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--only", default="", help="self-test: comma list of case numbers (1-based)")
    ap.add_argument("--alone", action="store_true", help="model alone, one shot per problem")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--run-name", default=time.strftime("alone_hard_%Y%m%d_%H%M%S"))
    a = ap.parse_args()

    if a.self_test:
        ok = True
        pick = [int(x) for x in a.only.split(",")] if a.only else range(1, len(SELF_TESTS) + 1)
        for i in pick:
            label, tid, src, expect = SELF_TESTS[i - 1]
            t0 = time.time()
            res = grade(BY_ID[tid], src)
            hit = res["status"] == expect
            ok &= hit
            extra = f"{res['peak_fraction']*100:.1f}% of peak" if "peak_fraction" in res else str(res.get("detail", ""))[:160]
            print(f"[{'OK ' if hit else 'BAD'}] {i}. {label:<42} -> {res['status']:<16} (expected {expect})  {extra}"
                  f"  [{time.time()-t0:.0f}s, gpu {res.get('gpu')}]", flush=True)
            if not hit: print("      detail:", json.dumps({k: v for k, v in res.items() if k != "report"})[:1500])
        print("\nHARD GRADER " + ("PASSED its self-test." if ok else "FAILED its self-test - fix before trusting scores."))
        return

    if a.alone:
        tasks = [BY_ID[x] for x in a.tasks.split(",")] if a.tasks else TASKS
        out_path = HERE / f"hard_results_{a.run_name}.jsonl"
        rows = []
        for seed in range(a.seeds):
            for t in tasks:
                text, finish, served, usage = ask_model(t, seed, a.temperature)
                code = extract_code(text)
                if not text.strip(): res = {"status": f"empty_response(finish={finish})"}
                elif code is None: res = {"status": "parse_error"}
                else: res = grade(t, code)
                row = {"seed": seed, "task": t["id"], **{k: v for k, v in res.items() if k != "report"},
                       "finish_reason": finish, "served_model": served,
                       "usage": usage.model_dump() if usage else None, "code": code, "output": text}
                rows.append(row)
                with open(out_path, "a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
                pf = f" {res['peak_fraction']*100:.1f}% of peak" if "peak_fraction" in res else ""
                print(f"seed={seed} {t['id']:<6} {res['status']}{pf}  {str(res.get('detail',''))[:150]}", flush=True)
        report(rows)

if __name__ == "__main__":
    main()
