"""
Validation for motion.py — accelerometer motion artifact detection (P2-2)

Quality gates:
    1. Method A (threshold): F1 ≥ 0.80 on synthetic scenario
    2. Method B (variance):  F1 ≥ 0.80 on synthetic scenario
    3. Both methods: precision ≥ 0.70 (not too many false positives)
    4. Both methods: recall ≥ 0.80 (don't miss real motion)
    5. Resting baseline: motion fraction < 5% on rest-only signal
    6. Dispatcher: ValueError on unknown method
    7. motion_fraction output: correct length and range [0, 1]
    8. motion_fraction detects high-motion windows correctly

Algorithm Engineering Discipline Rule 4:
    Reports precision AND recall, not just F1 alone.
    A method with high recall but low precision means too much good data
    is discarded; a method with high precision but low recall misses motion.

Run with:
    .venv/bin/python tests/test_motion.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.pipeline.motion import (
    svm_magnitude,
    detect_motion_threshold,
    detect_motion_variance,
    detect_motion,
    motion_fraction,
)

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)


# ── Synthetic accelerometer scenario ─────────────────────────────────────────
#
# 5-minute recording at 256 Hz.
# Three motion events with various characteristics:
#   Event 1: 30–40 s  (sustained walking, 2 Hz, high amplitude)
#   Event 2: 90–95 s  (brief arm movement, 4 Hz, moderate amplitude)
#   Event 3: 200–215 s (long exercise bout, 3 Hz, high amplitude)
# Rest: Gaussian noise ±0.2 m/s² on each axis (sensor noise floor).

FS = 256
DURATION_S = 300.0
N = int(DURATION_S * FS)

MOTION_EVENTS = [
    (30.0,  40.0),    # 10 s sustained
    (90.0,  95.0),    # 5 s brief
    (200.0, 215.0),   # 15 s long
]

rng = np.random.default_rng(77)

# Rest baseline: random noise + gravity on z-axis
ax = rng.normal(0.0, 0.20, N)
ay = rng.normal(0.0, 0.20, N)
az = 9.81 + rng.normal(0.0, 0.20, N)

# Ground-truth mask
truth = np.zeros(N, dtype=bool)

for (t_start, t_end) in MOTION_EVENTS:
    i0 = int(t_start * FS)
    i1 = int(t_end   * FS)
    truth[i0:i1] = True
    t = np.arange(i1 - i0) / FS
    # Multi-frequency sinusoidal motion: simulate walking/arm swing
    freq1 = rng.uniform(2.0, 4.0)
    freq2 = rng.uniform(3.0, 6.0)
    amp   = rng.uniform(3.0, 8.0)   # m/s²; well above threshold of 2.0
    ax[i0:i1] += amp * np.sin(2 * np.pi * freq1 * t)
    ay[i0:i1] += amp * 0.7 * np.sin(2 * np.pi * freq2 * t + 0.5)
    az[i0:i1] += amp * 0.5 * np.sin(2 * np.pi * (freq1 + 0.5) * t + 1.2)


# ── F1 helper ────────────────────────────────────────────────────────────────

def f1_precision_recall(pred: np.ndarray, truth: np.ndarray):
    pred  = pred.astype(bool)
    truth = truth.astype(bool)
    tp = int(np.sum(pred  &  truth))
    fp = int(np.sum(pred  & ~truth))
    fn = int(np.sum(~pred &  truth))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return f1, precision, recall, tp, fp, fn


# ── Run detection ─────────────────────────────────────────────────────────────

mask_a = detect_motion_threshold(ax, ay, az, FS, threshold_ms2=2.0)
mask_b = detect_motion_variance(ax, ay, az, FS, threshold_ms2=2.0)

f1_a, prec_a, rec_a, tp_a, fp_a, fn_a = f1_precision_recall(mask_a, truth)
f1_b, prec_b, rec_b, tp_b, fp_b, fn_b = f1_precision_recall(mask_b, truth)


# ── Resting baseline check ────────────────────────────────────────────────────

n_rest = int(20.0 * FS)    # first 20 s (no motion)
ax_rest = ax[:n_rest]
ay_rest = ay[:n_rest]
az_rest = az[:n_rest]

mask_a_rest = detect_motion_threshold(ax_rest, ay_rest, az_rest, FS, threshold_ms2=2.0)
mask_b_rest = detect_motion_variance(ax_rest, ay_rest, az_rest, FS, threshold_ms2=2.0)
rest_fp_a = float(np.mean(mask_a_rest))
rest_fp_b = float(np.mean(mask_b_rest))


# ── Print comparison table ────────────────────────────────────────────────────

print("=" * 70)
print("Motion artifact detection validation (P2-2)")
print(f"Signal: {DURATION_S:.0f} s, {FS} Hz, N={N} samples")
print(f"Motion events: {MOTION_EVENTS}")
print(f"Motion fraction (truth): {np.mean(truth)*100:.1f}%")
print("-" * 70)

print(f"\n{'Metric':<28} {'Method A (SVM-RMS)':>20} {'Method B (Rolling-Std)':>22}")
print("-" * 72)
for label, va, vb in [
    ("F1",              f1_a,   f1_b),
    ("Precision",       prec_a, prec_b),
    ("Recall",          rec_a,  rec_b),
    ("TP (samples)",    tp_a,   tp_b),
    ("FP (samples)",    fp_a,   fp_b),
    ("FN (samples)",    fn_a,   fn_b),
]:
    print(f"  {label:<26} {va:>20.4f} {vb:>22.4f}"
          if isinstance(va, float)
          else f"  {label:<26} {va:>20d} {vb:>22d}")

print(f"\nResting false-positive rate (first 20 s, truth=0):")
print(f"  Method A: {rest_fp_a*100:.2f}%")
print(f"  Method B: {rest_fp_b*100:.2f}%")


# ── Quality gate checks ───────────────────────────────────────────────────────

print()
print("Quality gate checks:")

check("Method A: F1 ≥ 0.80",
      f1_a >= 0.80,
      f"F1={f1_a:.4f}")
check("Method B: F1 ≥ 0.80",
      f1_b >= 0.80,
      f"F1={f1_b:.4f}")
check("Method A: precision ≥ 0.70",
      prec_a >= 0.70,
      f"precision={prec_a:.4f}")
check("Method B: precision ≥ 0.70",
      prec_b >= 0.70,
      f"precision={prec_b:.4f}")
check("Method A: recall ≥ 0.80",
      rec_a >= 0.80,
      f"recall={rec_a:.4f}")
check("Method B: recall ≥ 0.80",
      rec_b >= 0.80,
      f"recall={rec_b:.4f}")
check("Method A: resting FP rate < 5%",
      rest_fp_a < 0.05,
      f"{rest_fp_a*100:.2f}%")
check("Method B: resting FP rate < 5%",
      rest_fp_b < 0.05,
      f"{rest_fp_b*100:.2f}%")


# ── Error handling ────────────────────────────────────────────────────────────

print()
print("Error handling:")

raised = False
try:
    detect_motion(ax[:256], ay[:256], az[:256], FS, method="unknown")
except ValueError:
    raised = True
check("ValueError on unknown method", raised)


# ── motion_fraction checks ────────────────────────────────────────────────────

print()
print("motion_fraction checks:")

win_samples = int(10.0 * FS)   # 10 s windows
frac = motion_fraction(mask_a, win_samples)

expected_n_windows = len(mask_a) // win_samples
check(f"motion_fraction length = {expected_n_windows}",
      len(frac) == expected_n_windows,
      f"got {len(frac)}")
check("motion_fraction values in [0, 1]",
      bool(np.all((frac >= 0.0) & (frac <= 1.0))),
      f"range=[{frac.min():.3f}, {frac.max():.3f}]")

# Window containing event 1 (30–40 s, 10 s window → window index 3)
# Event spans window[3] (30–40 s); motion_fraction should be > 0.5
event1_win_idx = 3   # window at 30–40 s (0-indexed, 10 s step)
check(f"motion_fraction[{event1_win_idx}] (30–40 s event) > 0.5",
      frac[event1_win_idx] > 0.5,
      f"fraction={frac[event1_win_idx]:.3f}")

# First window (0–10 s, rest only) should be < 0.05
check("motion_fraction[0] (0–10 s, rest) < 0.05",
      frac[0] < 0.05,
      f"fraction={frac[0]:.3f}")


# ── Final summary ──────────────────────────────────────────────────────────────

print()
print("=" * 70)
n_pass = sum(results)
n_total = len(results)
print(f"Results: {n_pass}/{n_total} passed")

if n_pass == n_total:
    print("✓ Motion detection validated.")
    sys.exit(0)
else:
    print(f"✗ {n_total - n_pass} check(s) failed.")
    sys.exit(1)
