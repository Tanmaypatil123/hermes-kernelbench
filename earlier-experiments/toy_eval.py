"""
toy_eval.py - your first eval, from scratch, in one file.

What it does:
  1. TASKS   : 10 small "write this Python function" problems
  2. MODEL   : asks an LLM to solve each one (any OpenAI-compatible endpoint:
               vLLM, SGLang, OpenRouter, Together, etc.)
  3. GRADER  : runs hidden unit tests against the model's code -> pass/fail
  4. SCORE   : pass rate, repeated over several seeds, with a 95% interval
  5. LOGS    : every attempt saved to results.jsonl so you can read failures

Run:
  pip3 install openai          (Windows: pip / PowerShell $env:NAME = "...")
  export OPENAI_BASE_URL=http://localhost:8000/v1   # or https://openrouter.ai/api/v1
  export OPENAI_API_KEY=...                          # "EMPTY" for local vLLM
  export MODEL=Qwen/Qwen2.5-Coder-7B-Instruct        # whatever you are serving
  python3 toy_eval.py --seeds 3
"""

import argparse, json, math, os, random, re, statistics, subprocess, sys, tempfile

# ---------------------------------------------------------------- 1. TASKS
# Each task = a prompt the model sees + tests the model never sees.
TASKS = [
    {"id": "reverse",   "prompt": "Write `reverse(s)` that returns the string s reversed.",
     "tests": "assert reverse('abc')=='cba'\nassert reverse('')==''"},
    {"id": "is_prime",  "prompt": "Write `is_prime(n)` returning True if integer n is prime.",
     "tests": "assert is_prime(2) and is_prime(97)\nassert not is_prime(1) and not is_prime(0) and not is_prime(91)"},
    {"id": "fizz",      "prompt": "Write `fizz(n)` returning 'Fizz' if n%3==0, 'Buzz' if n%5==0, 'FizzBuzz' if both, else str(n).",
     "tests": "assert fizz(15)=='FizzBuzz' and fizz(9)=='Fizz' and fizz(10)=='Buzz' and fizz(7)=='7'"},
    {"id": "flatten",   "prompt": "Write `flatten(xs)` that flattens arbitrarily nested lists into one flat list.",
     "tests": "assert flatten([1,[2,[3,[4]]],5])==[1,2,3,4,5]\nassert flatten([])==[]"},
    {"id": "rmsnorm",   "prompt": "Write `rmsnorm(xs, eps=1e-6)` for a list of floats: x / sqrt(mean(x^2) + eps). Return a list.",
     "tests": "import math\nr=rmsnorm([3.0,4.0])\nassert all(abs(a-b)<1e-4 for a,b in zip(r,[3/math.sqrt(12.5),4/math.sqrt(12.5)]))"},
    {"id": "softmax",   "prompt": "Write `softmax(xs)` for a list of floats. It must not overflow for inputs like [1000, 1001].",
     "tests": "r=softmax([1000.0,1001.0])\nassert abs(sum(r)-1)<1e-6 and r[1]>r[0]"},
    {"id": "roman",     "prompt": "Write `to_roman(n)` converting 1<=n<=3999 to a Roman numeral string.",
     "tests": "assert to_roman(1994)=='MCMXCIV' and to_roman(4)=='IV' and to_roman(3999)=='MMMCMXCIX'"},
    {"id": "anagrams",  "prompt": "Write `group_anagrams(words)` returning a list of groups (lists) of anagrams; order does not matter.",
     "tests": "g=group_anagrams(['eat','tea','tan','ate','nat','bat'])\nassert sorted(sorted(x) for x in g)==[['ate','eat','tea'],['bat'],['nat','tan']]"},
    {"id": "lis",       "prompt": "Write `lis(xs)` returning the length of the longest strictly increasing subsequence.",
     "tests": "assert lis([10,9,2,5,3,7,101,18])==4 and lis([])==0 and lis([5,5,5])==1"},
    {"id": "matmul",    "prompt": "Write `matmul(A, B)` multiplying two matrices given as lists of lists. Raise ValueError if shapes mismatch.",
     "tests": "assert matmul([[1,2],[3,4]],[[5],[6]])==[[17],[39]]\ntry:\n    matmul([[1,2]],[[1,2]]); assert False\nexcept ValueError: pass"},
]

# ---------------------------------------------------------------- 2. MODEL
def ask_model(prompt, seed, temperature):
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_BASE_URL / OPENAI_API_KEY
    msg = (prompt + "\nReply with only a single ```python code block containing the function.")
    r = client.chat.completions.create(
        model=os.environ["MODEL"], seed=seed, temperature=temperature, max_tokens=4000,
        messages=[{"role": "user", "content": msg}],
    )
    c = r.choices[0]
    # finish_reason tells us WHY the model stopped: "stop" = finished normally,
    # "length" = ran out of tokens (often a reasoning model spent them all thinking).
    # r.model = the exact model the provider actually served (aliases like "-latest" can change!)
    return (c.message.content or ""), c.finish_reason, getattr(r, "model", None)

def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else None

# ---------------------------------------------------------------- 3. GRADER
def grade(code, tests, timeout=10):
    """Return (passed: bool, failure_reason: str|None)."""
    if code is None:
        return False, "parse_error"          # model didn't follow the format
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code + "\n\n" + tests + "\n")
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-X", "utf8", path], capture_output=True, timeout=timeout,
                           text=True, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)
    if p.returncode == 0:
        return True, None
    err = p.stderr
    if "AssertionError" in err: return False, "wrong_answer"
    if "SyntaxError" in err:    return False, "syntax_error"
    return False, "runtime_error"

# ---------------------------------------------------------------- 4. SCORE
def wilson(k, n, z=1.96):
    """95% confidence interval for a pass rate k/n."""
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--fake", action="store_true", help="no model; random answers, to test the plumbing")
    a = ap.parse_args()

    rows = []
    for seed in range(a.seeds):
        for t in TASKS:
            if a.fake:  # a "model" that is right ~60% of the time
                random.seed(hash((seed, t["id"])))
                text = "```python\ndef x(): pass\n```" if random.random() < 0.4 else "no code here"
                passed, reason = (random.random() < 0.6, None)
                if not passed: reason = random.choice(["wrong_answer", "parse_error", "timeout"])
                finish, served = "fake", "fake"
            else:
                text, finish, served = ask_model(t["prompt"], seed, a.temperature)
                if not text.strip():
                    # The model returned NOTHING. That is not a coding failure and not a
                    # parser bug - it is a harness/infra problem, so label it separately.
                    passed, reason = False, f"empty_response(finish={finish})"
                else:
                    passed, reason = grade(extract_code(text), t["tests"])
            rows.append({"seed": seed, "task": t["id"], "passed": passed, "reason": reason,
                         "finish_reason": finish, "served_model": served, "output": text})
            print(f"seed={seed} {t['id']:<10} {'PASS' if passed else 'FAIL ('+reason+')'}")

    with open("results.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    # ---------------------------------------------------------- 5. REPORT
    per_seed = [sum(r["passed"] for r in rows if r["seed"] == s) / len(TASKS) for s in range(a.seeds)]
    k, n = sum(r["passed"] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    print("\n=== RESULTS ===")
    print(f"model requested    : {os.environ.get('MODEL', 'fake')}")
    print(f"model(s) served    : {sorted(set(str(r['served_model']) for r in rows))}")
    print(f"pass rate per seed : {[round(x, 2) for x in per_seed]}")
    print(f"mean +- std         : {statistics.mean(per_seed):.2f} +- {statistics.pstdev(per_seed):.2f}")
    print(f"overall            : {k}/{n} = {k/n:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")
    print("(note: seeds of the same task are not independent, so this CI is a bit too narrow - a lesson for later)")
    reasons = {}
    for r in rows:
        if not r["passed"]: reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print(f"failure reasons    : {reasons}")
    always = [t['id'] for t in TASKS if all(r['passed'] for r in rows if r['task'] == t['id'])]
    print(f"pass^k (all seeds) : {len(always)}/{len(TASKS)} tasks -> {always}")

if __name__ == "__main__":
    main()
