"""
Run every tests/test_*.py validation script and summarise the outcome.

Each test is a standalone script that prints PASS/FAIL checks and exits non-zero
on failure. Scripts that need data absent from this checkout print "SKIPPED".

Run with:
    .venv/bin/python scripts/run_tests.py
"""

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main() -> int:
    rows, failed = [], 0
    for test in sorted((ROOT / "tests").glob("test_*.py")):
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(test)], cwd=ROOT,
                              capture_output=True, text=True)
        out = ANSI.sub("", proc.stdout + proc.stderr)
        n_pass = len(re.findall(r"\[PASS\s*\]", out))
        n_fail = len(re.findall(r"\[FAIL\s*\]", out))
        if proc.returncode != 0:
            status, failed = "FAIL", failed + 1
        elif "SKIPPED:" in out and n_pass == 0:
            status = "SKIPPED"
        else:
            status = "PASS"
        rows.append((test.name, status, n_pass, n_fail, time.time() - t0))
        if status == "FAIL":
            print(out[-3000:])

    print(f"{'test':<38} {'status':<8} {'checks':>11} {'time':>7}")
    for name, status, n_pass, n_fail, dt in rows:
        print(f"{name:<38} {status:<8} {n_pass:>4} pass {n_fail:>1} fail {dt:>6.1f}s")
    print(f"\n{len(rows) - failed}/{len(rows)} scripts passed or skipped; {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
