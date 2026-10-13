"""kernel_ledger.py (optional-skills/mlops/gpu-kernel-optimization): record, restore and verify.

The ledger's job is to make an agent finish on its best tested kernel, not its last edit. These
tests drive it the way the skill tells the agent to: wrap each test run, then restore + verify.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "optional-skills" / "mlops" / "gpu-kernel-optimization" / "scripts" / "kernel_ledger.py"

FAKE_TEST = """
import sys
src = open("solution.py").read()
if "BROKEN" in src:
    print("status : wrong_output")
    sys.exit(0)
print("status : correct")
print("speedup       : " + src.split("SPEED=")[1].split()[0] + "x")
"""


def _ledger(cwd, *args):
    return subprocess.run([sys.executable, str(LEDGER), *args], check=False, cwd=cwd, capture_output=True, text=True)


def _test_version(cwd, content, note):
    (cwd / "solution.py").write_text(content)
    return _ledger(cwd, "run", "--note", note, "--", sys.executable, "fake_test.py")


def test_restores_best_not_last(tmp_path):
    (tmp_path / "fake_test.py").write_text(FAKE_TEST)
    assert "v1 PASS metric=2.72" in _test_version(tmp_path, "SPEED=2.72", "fast").stdout
    slower = _test_version(tmp_path, "SPEED=1.05", "autotune").stdout
    assert "NOT better than v1" in slower
    assert "v3 FAIL" in _test_version(tmp_path, "BROKEN", "broken").stdout

    # the file now holds a failed version: verify must refuse it
    assert _ledger(tmp_path, "verify").returncode == 1
    assert _ledger(tmp_path, "restore").returncode == 0
    assert (tmp_path / "solution.py").read_text() == "SPEED=2.72"
    assert _ledger(tmp_path, "verify").returncode == 0


def test_verify_rejects_untested_edit(tmp_path):
    (tmp_path / "fake_test.py").write_text(FAKE_TEST)
    _test_version(tmp_path, "SPEED=2.0", "only version")
    (tmp_path / "solution.py").write_text("SPEED=2.0  # tidy comment added after the last test")
    r = _ledger(tmp_path, "verify")
    assert r.returncode == 1 and "untested" in r.stdout


def test_verify_warns_when_passing_but_not_best(tmp_path):
    (tmp_path / "fake_test.py").write_text(FAKE_TEST)
    _test_version(tmp_path, "SPEED=3.0", "best")
    _test_version(tmp_path, "SPEED=1.5", "worse")
    assert _ledger(tmp_path, "verify").returncode == 2


def test_parses_kernelbench_style_output(tmp_path):
    (tmp_path / "solution.py").write_text("x = 1")
    r = _ledger(tmp_path, "run", "--", sys.executable, "-c", "print('PASS'); print('peak_fraction: 0.0484')")
    assert "v1 PASS metric=0.0484" in r.stdout


def test_lower_is_better_metric(tmp_path):
    for ms in ("0.90", "0.50", "0.70"):
        (tmp_path / "solution.py").write_text(f"ms={ms}")
        _ledger(tmp_path, "run", "--metric-regex", r"time: ([0-9.]+) ms", "--lower-is-better", "--",
                sys.executable, "-c", f"print('PASS'); print('time: {ms} ms')")
    _ledger(tmp_path, "restore")
    assert (tmp_path / "solution.py").read_text() == "ms=0.50"


def test_hung_test_command_is_a_failure(tmp_path):
    (tmp_path / "solution.py").write_text("x = 1")
    r = _ledger(tmp_path, "run", "--timeout", "1", "--", sys.executable, "-c", "import time; time.sleep(30)")
    assert r.returncode == 124 and "v1 FAIL" in r.stdout and "timed out" in r.stdout


def test_quiet_keeps_scores_and_failures_and_logs_everything(tmp_path):
    (tmp_path / "solution.py").write_text("x = 1")
    noisy = "; ".join(f"print('noise line {i}')" for i in range(200))
    r = _ledger(tmp_path, "run", "--quiet", "--", sys.executable, "-c", noisy + "; print('PASS'); print('peak_fraction: 0.5')")
    assert "peak_fraction: 0.5" in r.stdout and "noise line 3" not in r.stdout
    assert "noise line 3" in (tmp_path / ".kernel_ledger" / "logs" / "v1.log").read_text()
    (tmp_path / "solution.py").write_text("x = 2")
    r = _ledger(tmp_path, "run", "--quiet", "--", sys.executable, "-c", noisy + "; print('FAIL: shape 3 wrong values')")
    assert "FAIL: shape 3 wrong values" in r.stdout and "v2 FAIL" in r.stdout


def test_restore_confirms_the_file_is_tested(tmp_path):
    (tmp_path / "fake_test.py").write_text(FAKE_TEST)
    _test_version(tmp_path, "SPEED=2.5", "good")
    (tmp_path / "solution.py").write_text("SPEED=9.9 untested")
    r = _ledger(tmp_path, "restore")
    assert "no further check needed" in r.stdout and _ledger(tmp_path, "verify").returncode == 0
