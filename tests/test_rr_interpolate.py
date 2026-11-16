"""
Validation for rr_interpolate.py — RR interpolation method comparison (P1-7)

Algorithm Engineering Discipline Rule 4: MUST report:
    - RMSE at original timestamps    → interpolation accuracy
    - n_nonphysiological             → boundary safety (V2 bug prevention)
    - max_error_ms                   → worst-case single-beat error

Quality gates:
    1. All methods: n_nonphysiological = 0  (no out-of-bounds values)
    2. All methods: RMSE < 10 ms            (< 1 beat interval error)
    3. Best method: RMSE < 2 ms            (< 8% of typical RMSSD=25ms)
    4. All methods: output length ≥ duration_s × target_fs × 0.95
    5. V2 regression: no extrapolation at boundaries (first/last value must
       be within physiological range)

References:
    Task Force ESC (1996) Circulation 93:1043-1065.

Run with:
    .venv/bin/python tests/test_rr_interpolate.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.pipeline.rr_interpolate import (
    compare_methods, interpolate_rr, rr_to_timestamps,
)

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
TARGET_FS = cfg["rr_interpolation"]["target_fs_hz"]       # 4 Hz
CLIP_MIN  = 300.0   # slightly wider than config physiological_min for clipping test
CLIP_MAX  = 2000.0

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)


# ── Synthetic RR sequence ──────────────────────────────────────────────────────

def make_rr(n_beats=300, mean_rr=800.0, rmssd=30.0, seed=42):
    rng = np.random.default_rng(seed)
    ar_coef = 0.5
    noise_std = rmssd / np.sqrt(2 * (1 - ar_coef**2))
    rr = np.zeros(n_beats)
    rr[0] = mean_rr
    for i in range(1, n_beats):
        rr[i] = mean_rr + ar_coef * (rr[i - 1] - mean_rr) + rng.normal(0, noise_std)
    return np.clip(rr, 450.0, 1400.0)


rr_normal = make_rr(n_beats=300, mean_rr=800.0, rmssd=30.0, seed=42)
ts_normal = rr_to_timestamps(rr_normal)

rr_high_hr = make_rr(n_beats=300, mean_rr=650.0, rmssd=25.0, seed=7)   # ~92 bpm
ts_high_hr = rr_to_timestamps(rr_high_hr)

CONFIGS = [
    {"name": "Cubic spline",   "method": "cubic_spline"},
    {"name": "Quadratic",      "method": "quadratic"},
    {"name": "Linear",         "method": "linear"},
]


# ── Test 1: Normal resting HR (70 bpm) ────────────────────────────────────────

print("=" * 72)
print("RR interpolation method comparison (P1-7)")
print(f"Config: target_fs={TARGET_FS} Hz, clip=[{CLIP_MIN}, {CLIP_MAX}] ms")
print("-" * 72)

print(f"\n[1] Normal resting HR (~70 bpm, mean_rr=800 ms, RMSSD=30 ms):")

comp = compare_methods(rr_normal, ts_normal, CONFIGS,
                       target_fs=TARGET_FS,
                       clip_min_ms=CLIP_MIN, clip_max_ms=CLIP_MAX)

print(f"\n{'Method':<18} {'N_uni':>7} {'RMSE ms':>9} {'MaxErr ms':>10} {'NonPhys':>8} {'Min ms':>8} {'Max ms':>8}")
print("-" * 75)
for r in comp:
    if "error" in r:
        print(f"{r['name']:<18}  ERROR: {r['error']}")
        continue
    print(
        f"{r['name']:<18} "
        f"{r['n_uniform']:>7d} "
        f"{r['rmse_ms']:>9.4f} "
        f"{r['max_error_ms']:>10.4f} "
        f"{r['n_nonphysiological']:>8d} "
        f"{r['rr_min_ms']:>8.2f} "
        f"{r['rr_max_ms']:>8.2f}"
    )

valid = [r for r in comp if "error" not in r]
duration_s = float(ts_normal[-1] - ts_normal[0])
min_expected_len = int(duration_s * TARGET_FS * 0.95)

print()
check(
    "All methods: n_nonphysiological = 0",
    all(r["n_nonphysiological"] == 0 for r in valid),
    f"max={max(r['n_nonphysiological'] for r in valid)}",
)
check(
    "All methods: RMSE < 10 ms",
    all(r["rmse_ms"] < 10.0 for r in valid),
    f"max={max(r['rmse_ms'] for r in valid):.4f} ms",
)
check(
    "Best method: RMSE < 2 ms",
    any(r["rmse_ms"] < 2.0 for r in valid),
    f"best={min(r['rmse_ms'] for r in valid):.4f} ms",
)
check(
    f"All methods: output length ≥ {min_expected_len} samples",
    all(r["n_uniform"] >= min_expected_len for r in valid),
    f"min={min(r['n_uniform'] for r in valid)}",
)


# ── Test 2: Higher HR (~92 bpm) ────────────────────────────────────────────────

print()
print("[2] Higher HR (~92 bpm, mean_rr=650 ms):")

comp_hhr = compare_methods(rr_high_hr, ts_high_hr, CONFIGS,
                            target_fs=TARGET_FS,
                            clip_min_ms=CLIP_MIN, clip_max_ms=CLIP_MAX)
valid_hhr = [r for r in comp_hhr if "error" not in r]

print(f"\n{'Method':<18} {'RMSE ms':>9} {'MaxErr ms':>10} {'NonPhys':>8}")
print("-" * 50)
for r in comp_hhr:
    if "error" in r:
        print(f"{r['name']:<18}  ERROR: {r['error']}")
        continue
    print(f"{r['name']:<18} {r['rmse_ms']:>9.4f} {r['max_error_ms']:>10.4f} {r['n_nonphysiological']:>8d}")

print()
check(
    "All methods: n_nonphysiological = 0 at 92 bpm",
    all(r["n_nonphysiological"] == 0 for r in valid_hhr),
    f"max={max(r['n_nonphysiological'] for r in valid_hhr)}",
)
check(
    "All methods: RMSE < 10 ms at 92 bpm",
    all(r["rmse_ms"] < 10.0 for r in valid_hhr),
    f"max={max(r['rmse_ms'] for r in valid_hhr):.4f} ms",
)


# ── Test 3: V2 regression — no boundary extrapolation ─────────────────────────

print()
print("[3] V2 regression: boundary extrapolation check:")

for cfg_item in CONFIGS:
    rr_uni, t_uni = interpolate_rr(
        rr_normal, ts_normal, cfg_item["method"],
        target_fs=TARGET_FS, clip_min_ms=CLIP_MIN, clip_max_ms=CLIP_MAX,
    )
    first_ok = CLIP_MIN <= float(rr_uni[0]) <= CLIP_MAX
    last_ok  = CLIP_MIN <= float(rr_uni[-1]) <= CLIP_MAX
    check(
        f"{cfg_item['name']}: first={rr_uni[0]:.1f} ms, last={rr_uni[-1]:.1f} ms in [{CLIP_MIN}, {CLIP_MAX}]",
        first_ok and last_ok,
    )


# ── Test 4: Edge cases ────────────────────────────────────────────────────────

print()
print("[4] Edge cases:")

# Minimum viable sequence (3 beats — minimum for quadratic)
rr_tiny = np.array([800.0, 810.0, 790.0])
ts_tiny = rr_to_timestamps(rr_tiny)
for cfg_item in CONFIGS:
    ok = True
    try:
        rr_u, t_u = interpolate_rr(rr_tiny, ts_tiny, cfg_item["method"],
                                    target_fs=TARGET_FS,
                                    clip_min_ms=CLIP_MIN, clip_max_ms=CLIP_MAX)
        ok = len(rr_u) > 0
    except Exception:
        ok = False
    check(f"{cfg_item['name']}: handles 3-beat sequence", ok)

# Dispatcher: unknown method raises ValueError
raised = False
try:
    interpolate_rr(rr_normal, ts_normal, "unknown")
except ValueError:
    raised = True
check("ValueError on unknown method", raised)


# ── Best method summary ────────────────────────────────────────────────────────

print()
best = min(valid, key=lambda r: r["rmse_ms"])
print(f"Best method by RMSE:")
print(f"  Method:     {best['name']}")
print(f"  RMSE:       {best['rmse_ms']:.4f} ms")
print(f"  Max error:  {best['max_error_ms']:.4f} ms")
print(f"  Output N:   {best['n_uniform']} samples ({best['n_uniform'] / TARGET_FS:.1f} s @ {TARGET_FS} Hz)")
print(f"  Range:      [{best['rr_min_ms']:.1f}, {best['rr_max_ms']:.1f}] ms")
print(f"  Non-phys:   {best['n_nonphysiological']}")


# ── Final summary ──────────────────────────────────────────────────────────────

print()
print("=" * 72)
n_pass = sum(results)
n_total = len(results)
print(f"Results: {n_pass}/{n_total} passed")

if n_pass == n_total:
    print("✓ RR interpolation validated.")
    sys.exit(0)
else:
    print(f"✗ {n_total - n_pass} check(s) failed.")
    sys.exit(1)
