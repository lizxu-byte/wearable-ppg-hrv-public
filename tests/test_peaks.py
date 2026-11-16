"""
Validation for peaks.py — peak detection method comparison (P1-5)

Algorithm Engineering Discipline Rule 4: MUST report all of:
    - Se (sensitivity), PPV, F1   → detection completeness
    - peak_timing_MAE_ms          → IBI/HRV accuracy
    - RMSSD error                 → downstream HRV impact

Quality gates:
    1. All methods: F1 ≥ 0.80 (clean signal)
    2. All methods: peak_timing_MAE < 10 ms (Allen 2007 criterion)
    3. Best method: F1 ≥ 0.95
    4. Best method: peak_timing_MAE < 5 ms
    5. Best method: RMSSD error < 10 ms
    6. Robustness: F1 ≥ 0.80 on noisy signal (SNR=10 dB)
    7. Robustness: F1 ≥ 0.80 on 90 bpm signal

Allen J (2007) Physiol Meas 28(3):R1-R39. ±100 ms tolerance window standard.

Run with:
    .venv/bin/python tests/test_peaks.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.validation.synthetic import generate
from src.pipeline.filter import bandpass_butterworth
from src.pipeline.peaks import compare_methods, detect_peaks, evaluate

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
FS  = cfg["sampling"]["fs"]

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"
WARN_MARK = "\033[93mWARN\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)

def warn(name, condition, detail=""):
    status = PASS_MARK if condition else WARN_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    # Warnings don't fail the suite — they are informational


# ── Generate test signals ──────────────────────────────────────────────────────

synth_clean = generate(
    duration_s=60.0, fs=FS,
    mean_hr_bpm=70.0, hr_variability_ms=20.0,
    noise_snr_db=40.0,           # near-clean signal
    baseline_amplitude=0.0,
    add_motion_artifact=False,
    seed=42,
)

synth_noisy = generate(
    duration_s=60.0, fs=FS,
    mean_hr_bpm=70.0, hr_variability_ms=20.0,
    noise_snr_db=10.0,           # realistic noisy signal
    baseline_amplitude=0.3, baseline_freq_hz=0.15,
    add_motion_artifact=False,
    seed=42,
)

synth_90bpm = generate(
    duration_s=60.0, fs=FS,
    mean_hr_bpm=90.0, hr_variability_ms=20.0,
    noise_snr_db=20.0,
    baseline_amplitude=0.3, baseline_freq_hz=0.2,
    add_motion_artifact=False,
    seed=42,
)

# Apply bandpass filter before peak detection (standard preprocessing)
# Recommended config from P1-4: 0.5–8 Hz, order 4, zero-phase
sig_clean_filt = bandpass_butterworth(synth_clean.signal, FS, 0.5, 8.0, 4, zero_phase=True)
sig_noisy_filt = bandpass_butterworth(synth_noisy.signal, FS, 0.5, 8.0, 4, zero_phase=True)
sig_90bpm_filt = bandpass_butterworth(synth_90bpm.signal, FS, 0.5, 8.0, 4, zero_phase=True)


# ── Method configurations ──────────────────────────────────────────────────────

CONFIGS = [
    {"name": "Elgendi 2013 (default)",  "method": "elgendi"},
    {"name": "Bishop 2018 (default)",   "method": "bishop"},
    {"name": "Prominence adaptive",     "method": "prominence"},
]


# ── Test 1: Clean signal (SNR=40 dB) ──────────────────────────────────────────
print("=" * 75)
print("Peak detection method comparison (P1-5)")
print(f"Signal: 60 s, 70 bpm, zero-phase Butterworth 0.5–8 Hz pre-applied")
print("-" * 75)

print("\n[1] Clean signal (SNR=40 dB, no baseline):")
comp_clean = compare_methods(sig_clean_filt, synth_clean.peak_indices, FS, CONFIGS)

print(f"\n{'Method':<32} {'Se':>6} {'PPV':>6} {'F1':>6} {'PkMAE ms':>10} {'RMSSD_err ms':>13} {'TP/FP/FN':>12}")
print("-" * 90)
for r in comp_clean:
    if "error" in r:
        print(f"{r['name']:<32}  ERROR: {r['error']}")
        continue
    print(
        f"{r['name']:<32} "
        f"{r['sensitivity']:>6.3f} "
        f"{r['ppv']:>6.3f} "
        f"{r['f1']:>6.3f} "
        f"{r['peak_timing_mae_ms']:>10.3f} "
        f"{r['rmssd_error_ms']:>13.3f} "
        f"{r['tp']:>4}/{r['fp']:>3}/{r['fn']:>3}"
    )

valid_clean = [r for r in comp_clean if "error" not in r]

print()
check(
    "All methods: F1 ≥ 0.80 on clean signal",
    all(r["f1"] >= 0.80 for r in valid_clean),
    f"min F1={min(r['f1'] for r in valid_clean):.3f}",
)
check(
    "All methods: peak_timing_MAE < 10 ms on clean signal",
    all(r["peak_timing_mae_ms"] < 10.0 for r in valid_clean),
    f"max MAE={max(r['peak_timing_mae_ms'] for r in valid_clean):.3f} ms",
)
check(
    "Best method: F1 ≥ 0.95 on clean signal",
    any(r["f1"] >= 0.95 for r in valid_clean),
    f"best F1={max(r['f1'] for r in valid_clean):.3f}",
)
check(
    "Best method: peak_timing_MAE < 5 ms on clean signal",
    any(r["peak_timing_mae_ms"] < 5.0 for r in valid_clean),
    f"best MAE={min(r['peak_timing_mae_ms'] for r in valid_clean):.3f} ms",
)
check(
    "Best method: RMSSD error < 10 ms on clean signal",
    any(r["rmssd_error_ms"] < 10.0 for r in valid_clean if np.isfinite(r["rmssd_error_ms"])),
    f"best RMSSD_err={min((r['rmssd_error_ms'] for r in valid_clean if np.isfinite(r['rmssd_error_ms'])), default=float('nan')):.3f} ms",
)


# ── Test 2: Noisy signal (SNR=10 dB + baseline) ────────────────────────────────
print()
print("[2] Noisy signal (SNR=10 dB, baseline_amp=0.3 @ 0.15 Hz):")
comp_noisy = compare_methods(sig_noisy_filt, synth_noisy.peak_indices, FS, CONFIGS)

print(f"\n{'Method':<32} {'Se':>6} {'PPV':>6} {'F1':>6} {'PkMAE ms':>10} {'RMSSD_err ms':>13} {'TP/FP/FN':>12}")
print("-" * 90)
for r in comp_noisy:
    if "error" in r:
        print(f"{r['name']:<32}  ERROR: {r['error']}")
        continue
    print(
        f"{r['name']:<32} "
        f"{r['sensitivity']:>6.3f} "
        f"{r['ppv']:>6.3f} "
        f"{r['f1']:>6.3f} "
        f"{r['peak_timing_mae_ms']:>10.3f} "
        f"{r['rmssd_error_ms']:>13.3f} "
        f"{r['tp']:>4}/{r['fp']:>3}/{r['fn']:>3}"
    )

valid_noisy = [r for r in comp_noisy if "error" not in r]

print()
check(
    "All methods: F1 ≥ 0.80 on noisy signal",
    all(r["f1"] >= 0.80 for r in valid_noisy),
    f"min F1={min(r['f1'] for r in valid_noisy):.3f}",
)


# ── Test 3: 90 bpm robustness ──────────────────────────────────────────────────
print()
print("[3] 90 bpm robustness check:")
comp_90bpm = compare_methods(sig_90bpm_filt, synth_90bpm.peak_indices, FS, CONFIGS)

valid_90bpm = [r for r in comp_90bpm if "error" not in r]
print(f"\n{'Method':<32} {'F1':>6} {'PkMAE ms':>10}")
print("-" * 55)
for r in comp_90bpm:
    if "error" in r:
        print(f"{r['name']:<32}  ERROR: {r['error']}")
        continue
    print(f"{r['name']:<32} {r['f1']:>6.3f} {r['peak_timing_mae_ms']:>10.3f}")

print()
check(
    "All methods: F1 ≥ 0.80 at 90 bpm",
    all(r["f1"] >= 0.80 for r in valid_90bpm),
    f"min F1={min(r['f1'] for r in valid_90bpm):.3f}",
)


# ── Test 4: Dispatcher + evaluate smoke tests ──────────────────────────────────
print()
print("[4] Dispatcher and evaluate unit tests:")

# Dispatcher: unknown method raises ValueError
raised = False
try:
    detect_peaks(sig_clean_filt, FS, "unknown_method")
except ValueError:
    raised = True
check("ValueError on unknown method", raised)

# evaluate: empty detected → Se=0, PPV=0
metrics_empty = evaluate(np.array([]), synth_clean.peak_indices, FS)
check(
    "evaluate: empty detected → sensitivity=0, ppv=0, F1=0",
    metrics_empty["sensitivity"] == 0.0 and metrics_empty["ppv"] == 0.0 and metrics_empty["f1"] == 0.0,
)

# evaluate: perfect detection (inject true peaks as detected)
metrics_perfect = evaluate(synth_clean.peak_indices, synth_clean.peak_indices, FS)
check(
    "evaluate: perfect detection → F1=1.0, peak_timing_MAE=0.0",
    metrics_perfect["f1"] == 1.0 and metrics_perfect["peak_timing_mae_ms"] == 0.0,
    f"F1={metrics_perfect['f1']:.3f}, MAE={metrics_perfect['peak_timing_mae_ms']:.3f} ms",
)

# evaluate: all false positives
rng = np.random.default_rng(99)
random_peaks = np.sort(rng.integers(0, len(sig_clean_filt), size=50))
metrics_fp = evaluate(random_peaks, synth_clean.peak_indices, FS)
warn(
    "evaluate: random peaks → low F1 (informational)",
    metrics_fp["f1"] < 0.8,
    f"F1={metrics_fp['f1']:.3f}",
)


# ── Best method summary ────────────────────────────────────────────────────────
print()
print("Best method on clean signal (by F1):")
best = max(valid_clean, key=lambda r: r["f1"])
print(f"  Method:            {best['name']}")
print(f"  F1:                {best['f1']:.4f}")
print(f"  Sensitivity:       {best['sensitivity']:.4f}")
print(f"  PPV:               {best['ppv']:.4f}")
print(f"  peak_timing_MAE:   {best['peak_timing_mae_ms']:.3f} ms")
print(f"  RMSSD_true:        {best['rmssd_true_ms']:.3f} ms")
print(f"  RMSSD_detected:    {best['rmssd_detected_ms']:.3f} ms")
print(f"  RMSSD_error:       {best['rmssd_error_ms']:.3f} ms")
print(f"  TP/FP/FN:          {best['tp']}/{best['fp']}/{best['fn']}")

print()
print("Error propagation (from peak-timing MAE → RMSSD estimate):")
mae = best["peak_timing_mae_ms"]
rmssd_prop = 2 * np.sqrt(2) * mae
print(f"  Theoretical RMSSD error ≈ 2√2 × peak_timing_MAE = 2√2 × {mae:.3f} = {rmssd_prop:.3f} ms")
print(f"  Actual RMSSD error: {best['rmssd_error_ms']:.3f} ms")


# ── Final summary ──────────────────────────────────────────────────────────────
print()
print("=" * 75)
n_pass  = sum(results)
n_total = len(results)
print(f"Results: {n_pass}/{n_total} passed")

if n_pass == n_total:
    print("✓ Peak detection validated.")
    sys.exit(0)
else:
    print(f"✗ {n_total - n_pass} check(s) failed.")
    sys.exit(1)
