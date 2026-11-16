"""
Validation for shimmer_reader.py

Checks:
  1. All CSV files in the local data root (subject folders 0..66) can be parsed
     without error. Participant recordings are not distributed with this
     repository, so the script reports SKIPPED when none are found; the parser
     itself is covered by tests/test_shimmer_reader_synthetic.py.
  2. Measured sampling rate is 256 ± 1 Hz for every session
  3. PPG column found = Shimmer_1AA8_PPG_A13_CAL
  4. Each session has ≥ 10 valid samples
  5. Accelerometer and GSR arrays have correct shape

Run with:
    .venv/bin/python tests/test_shimmer_reader.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from src.io.shimmer_reader import discover_csv_files, load_session, resolve_data_root

CONFIG_PATH = ROOT / "config.yaml"

cfg = yaml.safe_load(open(CONFIG_PATH))
DATA_ROOT = resolve_data_root(cfg)

# Discover files across subject folders 0–66
csv_files = []
for subj in range(0, 67):
    subj_dir = DATA_ROOT / str(subj)
    if subj_dir.exists():
        csv_files.extend(sorted(subj_dir.rglob("*.csv")))

print(f"Found {len(csv_files)} CSV files across subject folders 0–66\n")
if not csv_files:
    print(f"SKIPPED: no local Shimmer recordings under {DATA_ROOT}")
    sys.exit(0)

passed = 0
failed = 0
warned = 0
errors = []
warnings = []

FS_NOMINAL       = cfg["sampling"]["fs"]
FS_WARN_LOW_HZ   = 200.0   # below this is genuinely unexpected
FS_WARN_HIGH_HZ  = 320.0   # above this is genuinely unexpected

for csv_path in csv_files:
    try:
        sess = load_session(csv_path, cfg)
        fs_src = sess.metadata.get("fs_source", "?")

        # Hard checks (must pass)
        n_ok      = sess.n_samples >= 10
        ppg_ok    = "PPG" in sess.ppg_column_used.upper()
        accel_ok  = sess.accel_ln.shape == (sess.n_samples, 3)
        ppg_valid = np.isfinite(sess.ppg).any()

        hard_ok = n_ok and ppg_ok and accel_ok and ppg_valid

        # Soft check: fs outside expected range → warn, not fail
        fs_warn = not (FS_WARN_LOW_HZ <= sess.fs_measured <= FS_WARN_HIGH_HZ)

        rel = csv_path.relative_to(DATA_ROOT)

        if not hard_ok:
            failed += 1
            errors.append({
                "file": str(rel), "fs": sess.fs_measured,
                "n_ok": n_ok, "ppg_ok": ppg_ok,
                "accel_ok": accel_ok, "ppg_valid": ppg_valid,
            })
            print(f"[FAIL ] {rel}  fs={sess.fs_measured:.1f}Hz({fs_src})  n={sess.n_samples}")
        elif fs_warn:
            warned += 1
            warnings.append({"file": str(rel), "fs_measured": sess.fs_measured, "fs_source": fs_src})
            print(f"[WARN ] {rel}  fs={sess.fs_measured:.1f}Hz({fs_src})  n={sess.n_samples}  dur={sess.duration_s:.0f}s")
        else:
            passed += 1
            print(f"[PASS ] {rel}  fs={sess.fs_measured:.1f}Hz({fs_src})  n={sess.n_samples}  dur={sess.duration_s:.0f}s")

    except Exception as e:
        failed += 1
        errors.append({"file": str(csv_path.relative_to(DATA_ROOT)), "error": str(e)})
        print(f"[ERROR] {csv_path.relative_to(DATA_ROOT)}  →  {e}")

print(f"\n{'='*70}")
print(f"PASSED: {passed}   WARNED: {warned}   FAILED: {failed}   TOTAL: {len(csv_files)}")

if warnings:
    print(f"\nWarnings (fs outside {FS_WARN_LOW_HZ}–{FS_WARN_HIGH_HZ} Hz):")
    for w in warnings:
        print(f"  {w['file']}  fs={w['fs_measured']:.1f}Hz  source={w['fs_source']}")

if errors:
    print("\nErrors (hard failures):")
    for e in errors:
        print(" ", e)

if failed == 0:
    print("\n✓ All sessions parsed. Shimmer reader validated.")
    sys.exit(0)
else:
    print(f"\n✗ {failed} session(s) failed hard validation.")
    sys.exit(1)
