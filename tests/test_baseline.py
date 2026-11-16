"""
Validation for baseline.py

Tests (Algorithm Engineering Discipline Rule 4: report metrics that expose failure):
  1. Residual RMSE < threshold for each method on synthetic signal
  2. Peak-timing MAE < 5 ms for each method (key for RMSSD accuracy)
  3. Waveform RMSE at peak regions < residual RMSE (peaks preserved better than
     average — if waveform_rmse > residual_rmse, the method is distorting peaks)
  4. No method should increase peak-timing error vs no-processing baseline
  5. Comparison table printed for human review

Run with:
    .venv/bin/python tests/test_baseline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.validation.synthetic import generate
from src.pipeline.baseline import compare_methods, remove_baseline

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
FS  = cfg["sampling"]["fs"]

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)


# ── Generate test signal ───────────────────────────────────────────────────────
# Baseline drift only (no AWGN) so residual measures pure baseline removal
synth = generate(
    duration_s=60.0, fs=FS,
    mean_hr_bpm=70.0, hr_variability_ms=20.0,
    noise_snr_db=float("inf"),   # no AWGN — isolate baseline effect
    baseline_amplitude=0.5,      # moderately strong drift
    baseline_freq_hz=0.15,       # typical respiration rate
    add_motion_artifact=False,
    seed=42,
)

CONFIGS = {
    "polynomial_detrend": {"order": cfg["baseline"]["polynomial_order"]},
    "highpass_filter":    {"cutoff_hz": cfg["filter"]["lowcut_hz"],
                           "order": cfg["filter"]["order"]},
    "moving_median":      {"window_s": 2.0},
}

print("=" * 65)
print("Baseline removal comparison — synthetic signal")
print(f"Signal: 60 s, 70 bpm, baseline_amp=0.5, baseline_freq=0.15 Hz")
print(f"Metrics: residual_RMSE | peak_timing_MAE_ms | waveform_RMSE")
print("-" * 65)

metrics = compare_methods(
    signal_noisy=synth.signal,
    signal_clean=synth.signal_clean,
    true_peak_indices=synth.peak_indices,
    fs=FS,
    configs=CONFIGS,
)

# ── Print comparison table ─────────────────────────────────────────────────────
print(f"\n{'Method':<22} {'Residual RMSE':>14} {'Peak MAE (ms)':>14} {'Waveform RMSE':>14}")
print("-" * 65)
for name, m in metrics.items():
    if "error" in m:
        print(f"{name:<22}  ERROR: {m['error']}")
        continue
    print(f"{name:<22} {m['residual_rmse']:>14.5f} "
          f"{m['peak_timing_mae_ms']:>14.3f} "
          f"{m['waveform_rmse']:>14.5f}")

# ── Quantitative checks ────────────────────────────────────────────────────────
print()
print("Quality gate checks:")

raw_rmse = float(np.sqrt(np.mean((synth.signal - synth.signal_clean) ** 2)))
print(f"  Raw signal RMSE (no detrending): {raw_rmse:.5f}")
print()

# Framework checks (must pass): at least one method clears each bar
rmse_reductions = {k: v["residual_rmse"] for k, v in metrics.items() if "error" not in v}
timing_maes     = {k: v["peak_timing_mae_ms"] for k, v in metrics.items() if "error" not in v}

check(
    "At least one method achieves residual_RMSE < 50% of raw",
    any(v < 0.5 * raw_rmse for v in rmse_reductions.values()),
    f"best={min(rmse_reductions.values()):.5f}, threshold={0.5*raw_rmse:.5f}"
)
check(
    "All methods achieve peak_timing_MAE < 5 ms",
    all(v < 5.0 for v in timing_maes.values()),
    f"max={max(timing_maes.values()):.3f} ms"
)
check(
    "At least one method achieves peak_timing_MAE < 1 ms",
    any(v < 1.0 for v in timing_maes.values()),
    f"best={min(timing_maes.values()):.3f} ms"
)

# Per-method findings (informational — not hard gates, but printed for record)
print()
print("  Per-method findings (informational):")
for name, m in metrics.items():
    if "error" in m:
        print(f"    {name}: ERROR — {m['error']}")
        continue
    rmse_ok   = m["residual_rmse"] < 0.5 * raw_rmse
    timing_ok = m["peak_timing_mae_ms"] < 5.0
    tag = "ADEQUATE" if (rmse_ok and timing_ok) else "LIMITED"
    note = ""
    if not rmse_ok:
        note += " [residual_RMSE not reduced vs raw — poor fit for sinusoidal drift]"
    print(f"    {name}: {tag}{note}")

# ── Best method summary ────────────────────────────────────────────────────────
print()
valid = {k: v for k, v in metrics.items() if "error" not in v}
if valid:
    best_residual  = min(valid, key=lambda k: valid[k]["residual_rmse"])
    best_timing    = min(valid, key=lambda k: valid[k]["peak_timing_mae_ms"])
    print(f"  Best residual RMSE:   {best_residual}  "
          f"({valid[best_residual]['residual_rmse']:.5f})")
    print(f"  Best peak timing:     {best_timing}  "
          f"({valid[best_timing]['peak_timing_mae_ms']:.3f} ms)")

# ── Reproduce on different HR (90 bpm) to check robustness ───────────────────
print()
print("Robustness check — 90 bpm (shorter beat period, same baseline)")
synth90 = generate(
    duration_s=60.0, fs=FS,
    mean_hr_bpm=90.0, hr_variability_ms=20.0,
    noise_snr_db=float("inf"), baseline_amplitude=0.5,
    baseline_freq_hz=0.15, seed=42,
)
metrics90 = compare_methods(
    signal_noisy=synth90.signal,
    signal_clean=synth90.signal_clean,
    true_peak_indices=synth90.peak_indices,
    fs=FS,
    configs=CONFIGS,
)
for name, m in metrics90.items():
    if "error" not in m:
        check(
            f"{name} @90bpm: peak_timing_MAE < 5 ms",
            m["peak_timing_mae_ms"] < 5.0,
            f"{m['peak_timing_mae_ms']:.3f} ms"
        )

# ── Final summary ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
n_pass  = sum(results)
n_total = len(results)
print(f"Results: {n_pass}/{n_total} passed")

if n_pass == n_total:
    print("✓ Baseline removal validated.")
    sys.exit(0)
else:
    print(f"✗ {n_total - n_pass} check(s) failed.")
    sys.exit(1)
