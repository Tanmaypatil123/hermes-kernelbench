#!/usr/bin/env python3
"""kernel_ledger.py - remember every tested version of a kernel file, and finish on the best one.

Optimizing a kernel is a search: most edits make it slower or break it. Agents (and people) often
end on their LAST version instead of their BEST, or edit after the final test. The ledger snapshots
the file at every test, parses pass/fail and a score from the test output, and can restore and
verify the best passing version.

  python3 kernel_ledger.py run -- python3 test_kernel.py     # snapshot + test + parse + record
  python3 kernel_ledger.py record --status pass --metric 2.7  # record a result you got another way
  python3 kernel_ledger.py list                               # every version, best marked
  python3 kernel_ledger.py diff v2 v5                         # what changed between versions
  python3 kernel_ledger.py restore                            # put the best passing version back
  python3 kernel_ledger.py verify                             # exit 0 only if the file IS the best passing version

State lives in ./.kernel_ledger/ (index.jsonl + snapshots/). Standard library only.
"""
import argparse, difflib, hashlib, json, re, shutil, subprocess, sys, time
from pathlib import Path

LEDGER = Path(".kernel_ledger")
INDEX = LEDGER / "index.jsonl"
SNAPS = LEDGER / "snapshots"

# Output lines that mean the test passed / failed. First match in each list wins.
PASS_PATTERNS = [r"^\s*PASS\s*$", r"status\s*:\s*correct\b", r"\ball tests passed\b", r"^\s*OK\s*$"]
FAIL_PATTERNS = [r"^\s*FAIL", r"status\s*:\s*(?!correct\b)[a-z_]+", r"Traceback \(most recent call last\)",
                 r"\b(wrong_output|compile_error|runtime_error|import_error|forbidden_op)\b"]
# Score patterns, tried in order; the LAST occurrence of the first pattern that matches is used.
METRIC_PATTERNS = [r"peak_fraction\s*[:=]\s*([0-9.]+)", r"speedup\s*[:=]\s*([0-9.]+)",
                   r"([0-9.]+)\s*TFLOPS", r"gbps\s*[:=]\s*([0-9.]+)"]


def _load():
    if not INDEX.exists():
        return []
    return [json.loads(l) for l in INDEX.read_text(encoding="utf-8").splitlines() if l.strip()]


def _append(entry):
    SNAPS.mkdir(parents=True, exist_ok=True)
    with INDEX.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _snapshot(path, version):
    src = Path(path)
    if not src.exists():
        sys.exit(f"ledger: {path} does not exist - write it first")
    dst = SNAPS / f"{version}{src.suffix}"
    SNAPS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def _next_version(entries):
    return f"v{len(entries) + 1}"


def _best(entries, lower_is_better=None):
    passed = [e for e in entries if e["status"] == "pass"]
    if not passed:
        return None
    scored = [e for e in passed if e.get("metric") is not None]
    if not scored:
        return passed[-1]
    lower = lower_is_better if lower_is_better is not None else scored[-1].get("lower_is_better", False)
    return (min if lower else max)(scored, key=lambda e: e["metric"])


def _parse(output, metric_regex):
    lines = output.splitlines()
    failed = any(re.search(p, l, re.IGNORECASE) for p in FAIL_PATTERNS for l in lines)
    passed = any(re.search(p, l, re.IGNORECASE) for p in PASS_PATTERNS for l in lines)
    metric = None
    for pat in ([metric_regex] if metric_regex else METRIC_PATTERNS):
        hits = re.findall(pat, output, re.IGNORECASE)
        if hits:
            try:
                metric = float(hits[-1])
            except ValueError:
                pass
            break
    status = "pass" if passed and not failed else ("fail" if failed or not passed else "unknown")
    return status, metric


def _quiet(output, status):
    """The lines worth reading: score lines on a pass; the failure plus the last lines on a fail."""
    lines = output.splitlines()
    if status == "pass":
        keep = [l for l in lines if any(re.search(p, l, re.IGNORECASE) for p in PASS_PATTERNS + METRIC_PATTERNS)
                or re.search(r"\bms\b|gbps|tflops|status", l, re.IGNORECASE)]
        return "\n".join(keep[-40:]) + "\n"
    first = next((i for i, l in enumerate(lines) if any(re.search(p, l, re.IGNORECASE) for p in FAIL_PATTERNS)), None)
    head = lines[first:first + 15] if first is not None else []
    tail = [l for l in lines[-25:] if l not in head]
    return "\n".join(head + (["..."] if head and tail else []) + tail) + "\n"


def _fmt(e, best):
    m = "-" if e.get("metric") is None else f"{e['metric']:.4g}"
    star = "  <- best" if best and e["version"] == best["version"] else ""
    note = f"  {e['note']}" if e.get("note") else ""
    return f"{e['version']:>5}  {e['status']:<7} metric={m:<10} sha={e['sha']}{note}{star}"


def cmd_run(a):
    entries = _load()
    if not a.cmd:
        sys.exit("ledger: give the test command after --, e.g. run -- python3 test_kernel.py")
    version = _next_version(entries)
    snap = _snapshot(a.file, version)
    sha = _sha(a.file)
    t0 = time.time()
    try:
        p = subprocess.run(a.cmd, check=False, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=a.timeout)
        returncode, output = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        partial = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        returncode, output = 124, partial + f"\nledger: test command timed out after {a.timeout}s\n"
    status, metric = _parse(output, a.metric_regex)
    log = LEDGER / "logs" / f"{version}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(output, encoding="utf-8")
    shown = _quiet(output, status) if a.quiet else output
    print(shown, end="" if shown.endswith("\n") else "\n")
    if a.quiet:
        print(f"(full output: {log})")
    if returncode != 0 and status == "pass":
        status = "fail"
    entry = {"version": version, "file": a.file, "sha": sha, "snapshot": snap, "status": status,
             "metric": metric, "lower_is_better": a.lower_is_better, "note": a.note,
             "exit_code": returncode, "secs": round(time.time() - t0, 1), "t": time.time()}
    _append(entry)
    best = _best(entries + [entry], a.lower_is_better)
    bm = "-" if not best or best.get("metric") is None else f"{best['metric']:.4g}"
    print(f"\nledger: {version} {status.upper()} metric={metric}  |  best so far: "
          f"{best['version'] + ' metric=' + bm if best else 'none passing yet'}")
    if status == "pass" and best and best["version"] != version:
        print(f"ledger: {version} is NOT better than {best['version']}. Try another idea, or "
              f"`kernel_ledger.py restore` to go back to {best['version']}.")
    sys.exit(returncode)


def cmd_record(a):
    entries = _load()
    version = _next_version(entries)
    snap = _snapshot(a.file, version)
    entry = {"version": version, "file": a.file, "sha": _sha(a.file), "snapshot": snap, "status": a.status,
             "metric": a.metric, "lower_is_better": a.lower_is_better, "note": a.note, "t": time.time()}
    _append(entry)
    print(_fmt(entry, _best(entries + [entry], a.lower_is_better)))


def cmd_list(a):
    entries = _load()
    if not entries:
        print("ledger: empty - test your first version with `kernel_ledger.py run -- <test command>`")
        return
    best = _best(entries)
    for e in entries:
        print(_fmt(e, best))
    cur = _sha(entries[-1]["file"]) if Path(entries[-1]["file"]).exists() else None
    match = [e["version"] for e in entries if e["sha"] == cur]
    print(f"current {entries[-1]['file']}: " + (f"same as {', '.join(match)}" if match else "UNTESTED (changed since last test)"))


def cmd_diff(a):
    entries = {e["version"]: e for e in _load()}
    for v in (a.a, a.b):
        if v not in entries:
            sys.exit(f"ledger: no version {v}")
    x = Path(entries[a.a]["snapshot"]).read_text(encoding="utf-8").splitlines(keepends=True)
    y = Path(entries[a.b]["snapshot"]).read_text(encoding="utf-8").splitlines(keepends=True)
    sys.stdout.writelines(difflib.unified_diff(x, y, a.a, a.b))


def cmd_restore(a):
    entries = _load()
    target = next((e for e in entries if e["version"] == a.version), None) if a.version else _best(entries)
    if not target:
        sys.exit("ledger: nothing to restore - no passing version recorded")
    if target["status"] != "pass":
        print(f"ledger: warning, {target['version']} did not pass")
    shutil.copyfile(target["snapshot"], target["file"])
    print(f"ledger: restored {target['version']} (metric={target.get('metric')}) to {target['file']}")
    if target["status"] == "pass" and _sha(target["file"]) == target["sha"]:
        print(f"ledger: OK - {target['file']} is now the tested version {target['version']}; no further check needed.")


def cmd_verify(a):
    entries = _load()
    best = _best(entries)
    if not entries:
        sys.exit("ledger: FAIL - nothing was ever tested")
    path = a.file or entries[-1]["file"]
    cur = _sha(path)
    tested = [e for e in entries if e["sha"] == cur]
    if not tested:
        print(f"ledger: FAIL - {path} was changed after its last test; it is untested. "
              f"Test it, or `kernel_ledger.py restore` the best version ({best['version'] if best else 'none'}).")
        sys.exit(1)
    if not any(e["status"] == "pass" for e in tested):
        print(f"ledger: FAIL - {path} matches {tested[-1]['version']}, which did not pass.")
        sys.exit(1)
    if best and best["sha"] != cur:
        print(f"ledger: WARN - {path} passes ({tested[-1]['version']}, metric={tested[-1].get('metric')}) but "
              f"{best['version']} scored better (metric={best.get('metric')}). Run `kernel_ledger.py restore`.")
        sys.exit(2)
    print(f"ledger: OK - {path} is {tested[-1]['version']}, the best passing version (metric={tested[-1].get('metric')}).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    r = sub.add_parser("run", help="snapshot the file, run the test, record the result")
    r.add_argument("--file", default="solution.py")
    r.add_argument("--metric-regex", help="regex with one group for the score (default: peak_fraction, speedup, TFLOPS, gbps)")
    r.add_argument("--lower-is-better", action="store_true", help="e.g. when the score is a time in ms")
    r.add_argument("--note", default="", help="what you changed in this version")
    r.add_argument("--timeout", type=int, default=1800, help="seconds before the test command counts as hung")
    r.add_argument("--quiet", action="store_true",
                   help="print only score lines (pass) or the failure and last lines (fail); full output goes to .kernel_ledger/logs/")
    r.add_argument("cmd", nargs=argparse.REMAINDER)

    rc = sub.add_parser("record", help="record a result obtained outside `run`")
    rc.add_argument("--file", default="solution.py")
    rc.add_argument("--status", choices=["pass", "fail"], required=True)
    rc.add_argument("--metric", type=float)
    rc.add_argument("--lower-is-better", action="store_true")
    rc.add_argument("--note", default="")

    sub.add_parser("list", help="show every version")
    d = sub.add_parser("diff", help="diff two versions")
    d.add_argument("a"); d.add_argument("b")
    rs = sub.add_parser("restore", help="copy the best (or a given) passing version back")
    rs.add_argument("--version")
    v = sub.add_parser("verify", help="exit 0 only if the file is the best passing version")
    v.add_argument("--file")

    a = ap.parse_args()
    if a.cmd_name == "run" and a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    {"run": cmd_run, "record": cmd_record, "list": cmd_list, "diff": cmd_diff,
     "restore": cmd_restore, "verify": cmd_verify}[a.cmd_name](a)


if __name__ == "__main__":
    main()
