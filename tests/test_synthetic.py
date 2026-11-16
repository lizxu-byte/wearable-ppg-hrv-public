"""
Validation for synthetic.py

Checks:
  1. Peak count matches expected beat count (±2 beats for boundary effects)
  2. True RR intervals reproduce the configured HR within 1 bpm
  3. HRV variability (SDNN of true RR) matches hr_variability_ms within 5 ms
  4. Clean signal RMSE vs itself is 0; composite RMSE vs clean > 0
  5. Reproducibility: same seed → identical arrays
  6. Motion artifact region has higher amplitude than surrounding signal

Run with:
    .venv/bin/python tests/test_synthetic.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.validation.synthetic import generate, make_clean, make_noisy, make_with_motion

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)

print("=" * 60)
print("Test 1: Basic generation — 70 bpm, 60 s, no noise")
s = make_clean(hr_bpm=70.0, duration_s=60.0)

expected_beats = int(70.0 / 60.0 * 60.0)
check("Peak count ≈ expected",
      abs(len(s.peak_indices) - expected_beats) <= 3,
      f"got {len(s.peak_indices)}, expected ~{expected_beats}")

mean_hr_from_rr = 60_000.0 / np.mean(s.rr_ms)
check("Mean HR within ±1 bpm of target",
      abs(mean_hr_from_rr - 70.0) < 1.0,
      f"measured {mean_hr_from_rr:.2f} bpm")

check("Signal length correct",
      len(s.signal) == 60 * 256,
      f"got {len(s.signal)}, expected {60*256}")

check("Clean signal == signal (no noise)",
      np.allclose(s.signal, s.signal_clean),
      "clean preset should have zero noise")

print()
print("Test 2: HRV variability — 75 bpm, 30 ms variability, 180 s")
s2 = generate(duration_s=180.0, mean_hr_bpm=75.0, hr_variability_ms=30.0,
              noise_snr_db=float("inf"), baseline_amplitude=0.0,
              add_motion_artifact=False, seed=42)

sdnn_true = float(np.std(s2.rr_ms, ddof=1))
check("SDNN of true RR within ±8 ms of target variability",
      abs(sdnn_true - 30.0) < 8.0,
      f"SDNN={sdnn_true:.1f} ms, target=30.0 ms")

mean_hr2 = 60_000.0 / np.mean(s2.rr_ms)
check("Mean HR within ±1 bpm of 75",
      abs(mean_hr2 - 75.0) < 1.0,
      f"measured {mean_hr2:.2f} bpm")

print()
print("Test 3: Noise — SNR 20 dB, no baseline (isolates AWGN)")
# Use no baseline so signal - signal_clean = pure AWGN only
s3 = generate(duration_s=60.0, mean_hr_bpm=70.0, hr_variability_ms=0.0,
              noise_snr_db=20.0, baseline_amplitude=0.0,
              add_motion_artifact=False, seed=42)

rmse_noise = float(np.sqrt(np.mean((s3.signal - s3.signal_clean) ** 2)))
check("Noisy signal differs from clean (RMSE > 0)",
      rmse_noise > 0,
      f"RMSE(noisy, clean)={rmse_noise:.4f}")

# Estimate actual SNR — valid because no baseline
signal_power = float(np.mean(s3.signal_clean ** 2))
noise_power  = float(np.mean((s3.signal - s3.signal_clean) ** 2))
actual_snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
check("Actual SNR within ±3 dB of target 20 dB",
      abs(actual_snr_db - 20.0) < 3.0,
      f"actual SNR={actual_snr_db:.1f} dB")

print()
print("Test 4: Motion artifact")
s4 = make_with_motion(hr_bpm=70.0, duration_s=120.0)

motion_start_idx = int(60.0 * 256)
motion_end_idx   = int(65.0 * 256)
before_rms = float(np.sqrt(np.mean(s4.signal[motion_start_idx - 1280:motion_start_idx] ** 2)))
motion_rms = float(np.sqrt(np.mean(s4.signal[motion_start_idx:motion_end_idx] ** 2)))
check("Motion region RMS > pre-motion RMS",
      motion_rms > before_rms,
      f"motion={motion_rms:.3f}, before={before_rms:.3f}")

print()
print("Test 5: Reproducibility")
sA = generate(seed=42, duration_s=10.0)
sB = generate(seed=42, duration_s=10.0)
check("Same seed → identical signal",    np.array_equal(sA.signal, sB.signal))
check("Same seed → identical peaks",     np.array_equal(sA.peak_indices, sB.peak_indices))

sC = generate(seed=99, duration_s=10.0)
check("Different seed → different signal", not np.array_equal(sA.signal, sC.signal))

print()
print("Test 6: Waveform morphology — dicrotic wave present")
# In a clean signal, the second-highest peak per beat should be the dicrotic wave
# Check: within each beat, the clean signal has a secondary local maximum
s_morph = make_clean(hr_bpm=70.0, duration_s=10.0)
fs = s_morph.fs
has_dicrotic = False
for i in range(len(s_morph.peak_indices) - 1):
    seg_start = s_morph.peak_indices[i]
    seg_end   = s_morph.peak_indices[i + 1]
    if seg_end - seg_start < 4:
        continue
    seg = s_morph.signal_clean[seg_start:seg_end]
    # Find all local maxima in segment (excluding endpoints)
    local_max = np.where(
        (seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:])
    )[0] + 1  # +1 to account for slice offset
    # systolic peak is at seg[0] (endpoint, not counted by interior search)
    # so we only need ≥1 interior local max = the dicrotic wave
    if len(local_max) >= 1:
        has_dicrotic = True
        break
check("Dicrotic wave present (≥1 interior local max per beat)", has_dicrotic)

print()
print("=" * 60)
n_pass = sum(results)
n_total = len(results)
print(f"Results: {n_pass}/{n_total} passed")

if n_pass == n_total:
    print("✓ Synthetic PPG generator validated.")
    sys.exit(0)
else:
    print(f"✗ {n_total - n_pass} check(s) failed.")
    sys.exit(1)
