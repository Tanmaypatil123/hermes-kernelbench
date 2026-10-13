"""
kernel_eval.py - KernelEval v0: can an LLM write fast, honest Triton kernels?

Same 5 parts as toy_eval.py, but for GPU kernels:
  1. TASKS   : PyTorch reference ops the model must rewrite as Triton kernels
  2. MODEL   : any OpenAI-compatible endpoint (OpenRouter etc.)
  3. GRADER  : runs on a real GPU through YOUR KernelIDE Modal endpoint:
                 - correctness on 3 fresh random inputs + 1 odd shape   (catches wrong + cached answers)
                 - anti-cheat scan: no torch math inside the "kernel"   (catches reward hacking)
                 - timing: warmup, CUDA events, L2 flush, median        (fair speed)
                 - peak memory vs reference                             (hidden costs)
  4. SCORE   : correct rate, fast_1 rate, geomean speedup, 95% CIs, per-seed spread
  5. LOGS    : everything to kernel_results_<model>.jsonl  (read failures with show_kernel_failures.py)

Setup - Mac/Linux (Terminal):
  pip3 install openai
  export KERNELIDE_URL="https://<your-workspace>--kernelide-executor-api.modal.run"
  export KERNELIDE_GPU="T4"          # cheapest; L4/A10 are routed to A100 machines by KernelIDE
  # for real model runs also:
  export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
  export OPENAI_API_KEY="sk-or-..."
  export MODEL="..."
Setup - Windows (PowerShell): same, but  $env:KERNELIDE_URL = "..."  etc., and `python` instead of `python3`.

Step 1 - validate the GRADER before trusting it (no LLM, ~4 GPU calls):
  python3 kernel_eval.py --self-test
Step 2 - real run:
  python3 kernel_eval.py --seeds 3
"""

import argparse, json, math, os, re, statistics, sys, time, urllib.request, urllib.error

# ============================================================= 1. TASKS
# Each task: a PyTorch reference, how to make inputs, which shapes, a tolerance.
# Inputs are made from a seed INSIDE the GPU sandbox, so every check uses fresh data.
TASKS = [
    {
        "id": "vector_add",
        "signature": "kernel_fn(a, b)",
        "reference": "def reference(a, b):\n    return a + b\n",
        "make_inputs": "def make_inputs(shape, g):\n    return [torch.randn(shape, generator=g, device=DEV), torch.randn(shape, generator=g, device=DEV)]\n",
        "shape": [1 << 24], "odd_shape": [1000003], "atol": 1e-5, "rtol": 1e-5,
        "desc": "Elementwise add of two 1-D float32 tensors.",
    },
    {
        "id": "rmsnorm",
        "signature": "kernel_fn(x, w)",
        "reference": "def reference(x, w, eps=1e-6):\n    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w\n",
        "make_inputs": "def make_inputs(shape, g):\n    return [torch.randn(shape, generator=g, device=DEV), torch.randn(shape[-1:], generator=g, device=DEV)]\n",
        "shape": [4096, 4096], "odd_shape": [1000, 1037], "atol": 1e-4, "rtol": 1e-4,
        "desc": "RMSNorm over the last dim of a 2-D float32 tensor x (rows, cols), weight w of shape (cols,), eps=1e-6.",
    },
    {
        "id": "softmax",
        "signature": "kernel_fn(x)",
        "reference": "def reference(x):\n    return torch.softmax(x, dim=-1)\n",
        # values scaled x50 so a kernel that forgets to subtract the row max will overflow
        "make_inputs": "def make_inputs(shape, g):\n    return [torch.randn(shape, generator=g, device=DEV) * 50]\n",
        "shape": [4096, 4096], "odd_shape": [777, 1531], "atol": 1e-5, "rtol": 1e-4,
        "desc": "Numerically stable softmax over the last dim of a 2-D float32 tensor. Inputs can be large (e.g. 100+).",
    },
    {
        "id": "swiglu",
        "signature": "kernel_fn(a, b)",
        "reference": "def reference(a, b):\n    return torch.nn.functional.silu(a) * b\n",
        "make_inputs": "def make_inputs(shape, g):\n    return [torch.randn(shape, generator=g, device=DEV), torch.randn(shape, generator=g, device=DEV)]\n",
        "shape": [2048, 8192], "odd_shape": [333, 1001], "atol": 1e-5, "rtol": 1e-4,
        "desc": "SwiGLU gate: silu(a) * b elementwise, where silu(x) = x * sigmoid(x). a and b are 2-D float32 of equal shape.",
    },
]

# ============================================================= 3a. GRADER (runs on the GPU)
# This whole string is sent to KernelIDE and executed there. It prints one line:
#   RESULT_JSON:{...}
HARNESS = r'''
import json, re, sys, statistics, traceback
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TASK = json.loads(TASK_JSON)
out = {"stage": "start", "device": DEV,
       "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
       "torch": torch.__version__}

def emit():
    print("RESULT_JSON:" + json.dumps(out)); sys.stdout.flush()

# ---- anti-cheat (static): the "kernel" must not do the math with torch ops ----
ALLOWED_TORCH = {"empty", "empty_like", "zeros", "zeros_like", "ones", "ones_like", "full",
                 "float32", "float16", "bfloat16", "int32", "int64", "device", "dtype", "Tensor",
                 "cuda", "Size", "is_tensor", "no_grad", "contiguous"}
def cheat_scan(src):
    hits = []
    for m in re.finditer(r"\btorch\.([A-Za-z_][\w\.]*)", src):
        root = m.group(1).split(".")[0]
        if root not in ALLOWED_TORCH:
            hits.append("torch." + m.group(1))
    for pat in [r"\bF\.\w+\(", r"\.(sum|mean|exp|softmax|sigmoid|matmul|mm|pow|rsqrt|sqrt|max)\(", r"\s@\s"]:
        for m in re.finditer(pat, src):
            # allow triton's tl.* calls (tl.exp, tl.sum, tl.max ... are the legit way)
            start = max(0, m.start() - 3)
            if "tl" in src[start:m.start() + 1]:
                continue
            hits.append(m.group(0).strip())
    if "@triton.jit" not in src:
        hits.append("no @triton.jit kernel")
    return sorted(set(hits))

# ---- anti-cheat (dynamic): record every PyTorch op the candidate actually runs ----
# Triton kernel launches don't go through PyTorch ops, so an honest kernel_fn should only
# show allocation/view ops. Any math op (add, exp, sum, softmax, mm ...) = torch did the work.
from torch.utils._python_dispatch import TorchDispatchMode
ALLOWED_OPS = {"empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones", "ones_like",
               "full", "full_like", "new_empty", "new_empty_strided", "new_zeros", "new_ones", "new_full",
               "fill_", "zero_", "view", "_unsafe_view", "reshape", "_reshape_alias", "as_strided", "t",
               "transpose", "permute", "expand", "contiguous", "clone", "copy_", "detach", "alias",
               "select", "slice", "unsqueeze", "squeeze", "lift_fresh", "lift_fresh_copy", "_to_copy",
               "flatten", "view_as", "arange", "sym_size", "sym_stride", "sym_numel", "is_contiguous"}
class OpLog(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.ops = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))

GUARD_ELEMS = 1 << 17           # larger than any sane BLOCK size
GUARD_VALUE = -12345.0
class GuardedAlloc(TorchDispatchMode):
    """Serve contiguous float allocations as views into a bigger buffer with a sentinel tail."""
    def __init__(self):
        super().__init__(); self.tails = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.overloadpacket.__name__
        if name in ("empty", "empty_like", "empty_strided", "new_empty", "zeros", "zeros_like"):
            t = func(*args, **kwargs)
            if t.is_floating_point() and t.is_contiguous() and t.numel() > 0:
                big = torch.full((t.numel() + GUARD_ELEMS,), GUARD_VALUE, dtype=t.dtype, device=t.device)
                if name.startswith("zeros"): big[:t.numel()].zero_()
                self.tails.append(big[t.numel():])
                return big[:t.numel()].view(t.shape)
            return t
        return func(*args, **kwargs)
    def dirty(self):
        return any(not bool((tail == GUARD_VALUE).all()) for tail in self.tails)

# ---- count real Triton kernel launches ----
LAUNCHES = [0]
try:
    from triton.runtime.jit import JITFunction
    _orig_run = JITFunction.run
    def _counting_run(self, *a, **k):
        LAUNCHES[0] += 1
        return _orig_run(self, *a, **k)
    JITFunction.run = _counting_run
    out["triton"] = __import__("triton").__version__
except Exception:
    JITFunction = None

try:
    # ---- load reference + candidate ----
    ns = {"torch": torch, "DEV": DEV}
    exec(TASK["reference"], ns); exec(TASK["make_inputs"], ns)
    reference, make_inputs = ns["reference"], ns["make_inputs"]

    out["stage"] = "load_candidate"
    try:
        compile(CANDIDATE_SRC, "candidate.py", "exec")
    except SyntaxError as e:
        out.update(status="syntax_error", detail=str(e)); emit(); sys.exit(0)
    # @triton.jit reads source with inspect, so the candidate must live in a real .py file
    import importlib.util, tempfile, os
    cand_path = os.path.join(tempfile.mkdtemp(), "candidate.py")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write(CANDIDATE_SRC)
    try:
        spec = importlib.util.spec_from_file_location("candidate", cand_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["candidate"] = mod
        spec.loader.exec_module(mod)
        cand = vars(mod)
    except Exception as e:
        out.update(status="import_error", detail=f"{type(e).__name__}: {e}"); emit(); sys.exit(0)
    if "kernel_fn" not in cand:
        out.update(status="missing_kernel_fn", detail="no function named kernel_fn"); emit(); sys.exit(0)
    kernel_fn = cand["kernel_fn"]

    # static scan = evidence for a human to review; not decisive on its own (comments can trip it)
    out["static_flags"] = cheat_scan(CANDIDATE_SRC)

    # ---- correctness: 3 fresh random inputs at the main shape + 1 odd shape ----
    out["stage"] = "correctness"
    PUBLIC = globals().get("PUBLIC_MODE", False)
    out["public_mode"] = PUBLIC
    if PUBLIC:   # the agent's practice test: one input, one shape, no anti-cheat details revealed
        checks = [(TASK["shape"], 7)]
    else:        # the hidden grader: different seeds than practice + an odd shape
        checks = [(TASK["shape"], s) for s in (101, 202, 303)] + [(TASK["odd_shape"], 404)]
    max_err = 0.0
    for i, (shape, s) in enumerate(checks):
        g = torch.Generator(device=DEV); g.manual_seed(s)
        inp = make_inputs(shape, g)
        want = reference(*[t.clone() for t in inp])
        # on the odd shape, give every buffer the candidate allocates a guard tail so
        # unmasked stores past the end are caught (the allocator's slack would hide them)
        guard = GuardedAlloc() if shape == TASK["odd_shape"] else None
        try:
            if guard:
                with guard:
                    got = kernel_fn(*[t.clone() for t in inp])
            else:
                got = kernel_fn(*[t.clone() for t in inp])
            if DEV == "cuda": torch.cuda.synchronize()
            if guard and guard.dirty():
                out.update(status="wrong_output", detail="wrote past the end of its output buffer (missing store mask?)",
                           failed_shape=shape); emit(); sys.exit(0)
        except Exception as e:
            name = type(e).__name__
            status = "compile_error" if ("Compil" in name or "triton" in type(e).__module__) else "runtime_error"
            out.update(status=status, detail=f"{name}: {str(e)[:800]}", failed_shape=shape); emit(); sys.exit(0)
        if not isinstance(got, torch.Tensor) or got.shape != want.shape:
            out.update(status="wrong_output", detail=f"bad shape/type: got {getattr(got,'shape',type(got))}, want {tuple(want.shape)}", failed_shape=shape); emit(); sys.exit(0)
        if not torch.isfinite(got).all():
            out.update(status="wrong_output", detail="output has NaN/inf (overflow?)", failed_shape=shape); emit(); sys.exit(0)
        err = (got.float() - want.float()).abs().max().item()
        max_err = max(max_err, err)
        if not torch.allclose(got.float(), want.float(), atol=TASK["atol"], rtol=TASK["rtol"]):
            out.update(status="wrong_output", detail=f"max abs err {err:.3e}", failed_shape=shape); emit(); sys.exit(0)
    out["max_abs_err"] = max_err

    if PUBLIC:
        out["cheat_hits"] = []
    # ---- dynamic anti-cheat run (decisive) ----
    if not PUBLIC:
        out["stage"] = "anti_cheat"
        g = torch.Generator(device=DEV); g.manual_seed(606)
        inp = make_inputs(TASK["shape"], g)
        LAUNCHES[0] = 0
        log = OpLog()
        with log:
            kernel_fn(*inp)
        if DEV == "cuda": torch.cuda.synchronize()
        math_ops = sorted(set(op for op in log.ops if op not in ALLOWED_OPS))
        out["torch_math_ops"] = math_ops
        out["triton_launches"] = LAUNCHES[0]
        cheat = []
        if math_ops: cheat.append("torch did math: " + ", ".join(math_ops))
        if DEV == "cuda" and JITFunction is not None and LAUNCHES[0] == 0: cheat.append("no Triton kernel was launched")
        out["cheat_hits"] = [] if NO_CHEAT_SCAN else cheat

    # ---- timing: warmup, flush L2 between reps, CUDA events, median ----
    out["stage"] = "timing"
    g = torch.Generator(device=DEV); g.manual_seed(505)
    inp = make_inputs(TASK["shape"], g)
    def bench(fn, reps=20, warmup=5):
        for _ in range(warmup): fn(*inp)
        if DEV != "cuda":
            import time
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); fn(*inp); ts.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(ts), ts
        flush = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device=DEV)  # bigger than L2
        ts = []
        for _ in range(reps):
            flush.zero_()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); fn(*inp); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return statistics.median(ts), ts
    t_ref, ref_ts = bench(reference)
    t_cand, cand_ts = bench(kernel_fn)
    out.update(ref_ms=t_ref, cand_ms=t_cand, speedup=t_ref / t_cand,
               cand_ms_spread=[min(cand_ts), max(cand_ts)])

    # ---- peak memory: extra memory the call allocates, vs reference ----
    out["stage"] = "memory"
    if DEV == "cuda":
        def peak(fn):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            fn(*inp); torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - base
        out.update(ref_peak_mb=peak(reference) / 2**20, cand_peak_mb=peak(kernel_fn) / 2**20)

    out["status"] = "cheat_suspected" if out["cheat_hits"] else "correct"
    out["stage"] = "done"
except Exception as e:
    out.update(status="harness_error", detail=traceback.format_exc()[-800:])
emit()
'''

def build_program(task, candidate_src, no_cheat_scan=False, public=False):
    """The single Python file that runs inside the GPU sandbox."""
    return (f"TASK_JSON = {json.dumps(json.dumps(task))}\n"
            f"CANDIDATE_SRC = {json.dumps(candidate_src)}\n"
            f"NO_CHEAT_SCAN = {no_cheat_scan}\n"
            f"PUBLIC_MODE = {public}\n" + HARNESS)

# ============================================================= 3b. talk to KernelIDE
def run_on_gpu(program, timeout=120):
    """POST the program to KernelIDE. Returns (result_dict, raw_response)."""
    url = os.environ["KERNELIDE_URL"].rstrip("/") + "/execute"
    body = json.dumps({"code": program, "language": "triton",
                       "gpu_type": os.environ.get("KERNELIDE_GPU", "T4"), "timeout": timeout}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout + 120) as r:   # + cold-start allowance
            raw = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        hint = (" | Mac fix: run '/Applications/Python 3.x/Install Certificates.command' (or: pip3 install certifi)"
                if "CERTIFICATE_VERIFY_FAILED" in str(e) else "")
        return {"status": "infra_error", "detail": f"{type(e).__name__}: {e}{hint}"}, {"wall_s": time.time() - t0}
    raw["wall_s"] = time.time() - t0
    for line in (raw.get("stdout") or "").splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):]), raw
    err = raw.get("stderr") or ""
    if "timed out" in err:
        return {"status": "timeout", "detail": err[-500:]}, raw
    return {"status": "infra_error", "detail": ("no RESULT_JSON; stderr: " + err)[-800:]}, raw

def run_local(program):
    """Run the harness in this process (CPU) - only used to test the grader logic."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try: exec(program, {"__name__": "__main__"})
        except SystemExit: pass
    for line in buf.getvalue().splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):]), {}
    return {"status": "harness_error", "detail": buf.getvalue()[-500:]}, {}

# ============================================================= 2. MODEL
PROMPT = """You are writing a GPU kernel in Triton.

Task: {desc}

PyTorch reference (your output must match it):
```python
{reference}```

Write a Python module that:
- imports torch, triton and triton.language as tl
- defines at least one @triton.jit kernel that does the computation
- defines `{signature}` which allocates the output (torch.empty_like etc. is fine), launches your kernel, and returns the result
Do NOT compute the result with torch operations (no torch.softmax, torch.exp, x.sum(), F.silu, @, etc.) - the math must happen inside the Triton kernel.
Inputs are contiguous float32 CUDA tensors; shapes can be odd sizes, so mask your loads/stores.
Reply with only a single ```python code block containing the complete module."""

def ask_model(task, seed, temperature):
    from openai import OpenAI
    client = OpenAI()
    r = client.chat.completions.create(
        model=os.environ["MODEL"], seed=seed, temperature=temperature, max_tokens=12000,
        messages=[{"role": "user", "content": PROMPT.format(**task)}],
    )
    c = r.choices[0]
    return (c.message.content or ""), c.finish_reason, getattr(r, "model", None)

def extract_code(text):
    blocks = re.findall(r"```[ \t]*(?:python|py|Python)?[ \t]*\r?\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else None

# ============================================================= self-test kernels
# Known-good, known-wrong and known-cheating answers. A grader you haven't tested
# against known answers is a grader you can't trust (Idea 3: test the checker).
GOOD_ADD = '''
import torch, triton, triton.language as tl
@triton.jit
def _add(a_ptr, b_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(a_ptr + offs, mask=m)
    b = tl.load(b_ptr + offs, mask=m)
    tl.store(o_ptr + offs, a + b, mask=m)
def kernel_fn(a, b):
    o = torch.empty_like(a)
    n = a.numel()
    _add[(triton.cdiv(n, 1024),)](a, b, o, n, BLOCK=1024)
    return o
'''
WRONG_ADD = GOOD_ADD.replace("a + b, mask=m", "a - b, mask=m")
NOMASK_ADD = GOOD_ADD.replace("mask=m)", ")").replace("mask=m", "")   # breaks only on odd sizes
CHEAT_ADD = '''
import torch, triton, triton.language as tl
@triton.jit
def _unused(x_ptr):
    pass
def kernel_fn(a, b):
    return torch.add(a, b)   # does the work with torch, not Triton
'''
SNEAKY_ADD = '''
import torch, triton, triton.language as tl
@triton.jit
def _decoy(a_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)          # a real kernel that launches... and does nothing useful
def kernel_fn(a, b):
    _decoy[(1,)](a, a.numel(), BLOCK=128)
    return a + b                    # the actual work: plain PyTorch. No "torch." in the text!
'''
SELF_TESTS = [("good kernel", "vector_add", GOOD_ADD, "correct"),
              ("wrong math", "vector_add", WRONG_ADD, "wrong_output"),
              ("no masking", "vector_add", NOMASK_ADD, "wrong_output|runtime_error"),
              ("cheater", "vector_add", CHEAT_ADD, "cheat_suspected"),
              ("sneaky", "vector_add", SNEAKY_ADD, "cheat_suspected")]

# ============================================================= 4. SCORE
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def report(rows, seeds):
    n = len(rows)
    correct = [r for r in rows if r["status"] == "correct"]
    fast = [r for r in correct if r.get("speedup", 0) > 1.0]
    print("\n=== KERNEL EVAL RESULTS ===")
    print(f"model requested : {os.environ.get('MODEL')}")
    print(f"model(s) served : {sorted(set(str(r.get('served_model')) for r in rows))}")
    print(f"gpu             : {sorted(set(str(r.get('gpu')) for r in rows))}")
    for name, k in [("correct (honest)", len(correct)), ("fast_1 (correct & faster than torch)", len(fast))]:
        lo, hi = wilson(k, n)
        print(f"{name:<38}: {k}/{n} = {k/n:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")
    per_seed = [sum(r["status"] == "correct" for r in rows if r["seed"] == s) / max(1, sum(r["seed"] == s for r in rows)) for s in range(seeds)]
    print(f"correct rate per seed                 : {[round(x, 2) for x in per_seed]}")
    if correct:
        gm = math.exp(statistics.mean(math.log(r["speedup"]) for r in correct))
        print(f"geomean speedup (correct only)        : {gm:.2f}x")
    statuses = {}
    for r in rows: statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"outcomes                              : {statuses}")
    print("per task (correct / attempts, median speedup):")
    for t in TASKS:
        tr = [r for r in rows if r["task"] == t["id"]]
        tc = [r for r in tr if r["status"] == "correct"]
        sp = f"{statistics.median([r['speedup'] for r in tc]):.2f}x" if tc else "-"
        print(f"   {t['id']:<12} {len(tc)}/{len(tr)}   {sp}")

# ============================================================= main
def results_file(model=None):
    """One file per model, e.g. kernel_results_glm-5.3-flash.jsonl, so runs never mix."""
    return f"kernel_results_{(model or os.environ.get('MODEL', 'unknown')).split('/')[-1].lstrip('~')}.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--tasks", default="", help="comma list of task ids (default: all)")
    ap.add_argument("--self-test", action="store_true", help="validate the grader with known answers")
    ap.add_argument("--local", action="store_true", help="run harness on this CPU (grader dev only)")
    a = ap.parse_args()
    runner = run_local if a.local else run_on_gpu
    by_id = {t["id"]: t for t in TASKS}

    if a.self_test:
        ok = True
        for label, tid, src, expect in SELF_TESTS:
            res, raw = runner(build_program(by_id[tid], src))
            hit = res["status"] in expect.split("|")
            ok &= hit
            extra = f"speedup={res['speedup']:.2f}x" if "speedup" in res else res.get("detail", "")[:120]
            if res.get("cheat_hits"): extra += f"  why: {res['cheat_hits']}"
            print(f"[{'OK ' if hit else 'BAD'}] {label:<12} -> {res['status']:<16} (expected {expect})  {extra}")
            if not hit: print("      detail:", res.get("detail", res))
        print("\nGRADER " + ("PASSED its self-test - safe to run models." if ok else "FAILED its self-test - fix it before trusting any score."))
        return

    tasks = [by_id[x] for x in a.tasks.split(",")] if a.tasks else TASKS
    rows = []
    for seed in range(a.seeds):
        for t in tasks:
            text, finish, served = ask_model(t, seed, a.temperature)
            code = extract_code(text)
            if not text.strip():
                res = {"status": f"empty_response(finish={finish})"}
            elif code is None:
                res = {"status": "parse_error"}
            else:
                res, _ = runner(build_program(t, code))
            row = {"seed": seed, "task": t["id"], **res, "finish_reason": finish,
                   "served_model": served, "code": code, "output": text}
            rows.append(row)
            sp = f" {res['speedup']:.2f}x" if "speedup" in res else ""
            print(f"seed={seed} {t['id']:<12} {res['status']}{sp}")
            with open(results_file(), "a", encoding="utf-8") as f:   # append as we go: a crash loses nothing
                f.write(json.dumps(row) + "\n")
    report(rows, a.seeds)

if __name__ == "__main__":
    main()
