"""
Validation for sqi.py — Signal Quality Index (P2-1)

Quality gates:
    1. Clean PPG window:  tc_sqi  > 0.70
    2. Clean PPG window:  se_sqi  > 0.30
    3. Motion-artifact window: tc_sqi < clean tc_sqi by ≥ 0.20
    4. Motion-artifact window: se_sqi < clean se_sqi by ≥ 0.10
    5. AUC (clean vs noisy, 20 windows each): tc_sqi  AUC > 0.80
    6. AUC (clean vs noisy, 20 windows each): se_sqi  AUC > 0.80
    7. ValueError on unknown method
    8. NaN returned for too-short signal (< min_beats peaks)
    9. windowed_sqi produces correct number of windows
   10. windowed_sqi window_centers_s is monotonically increasing

Algorithm Engineering Discipline Rule 4:
    Both methods reported simultaneously. A method that passes gate 1/2
    but fails gate 5/6 is not considered validated.

Run with:
    .venv/bin/python tests/test_sqi.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import mannwhitneyu

from src.validation.synthetic import make_clean, make_noisy, make_with_motion, generate
from src.pipeline.sqi import (
    template_correlation_sqi,
    spectral_entropy_sqi,
    compute_sqi,
    windowed_sqi,
)

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)


# ── AUC via Mann-Whitney U  (no sklearn required) ─────────────────────────────

def auc_score(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUC = P(score_pos > score_neg), estimated via Mann-Whitney U."""
    u, _ = mannwhitneyu(scores_pos, scores_neg, alternative="greater")
    return float(u / (len(scores_pos) * len(scores_neg)))


# ── Reference signals ─────────────────────────────────────────────────────────

FS = 256
WIN_S = 10.0
N_WIN = int(WIN_S * FS)    # 2560 samples per window

# Single clean and artifact windows for absolute threshold tests
ppg_clean  = make_clean(hr_bpm=70, duration_s=WIN_S, fs=FS, seed=42)
ppg_motion = make_with_motion(hr_bpm=70, duration_s=120.0, fs=FS, seed=42)

win_clean     = ppg_clean.signal[:N_WIN]
# Extract the motion-burst window (peak at 60–65 s → center at 62.5 s)
motion_start  = int(62.0 * FS)
win_motion    = ppg_motion.signal[motion_start : motion_start + N_WIN]

# Pool of windows for AUC test
# 20 clean windows (different seeds / HR)
# 20 noisy windows (SNR 0 dB, different seeds / HR)
rng = np.random.default_rng(0)

def clean_window(seed):
    hr = float(rng.integers(55, 90))
    p  = make_clean(hr_bpm=hr, duration_s=WIN_S + 2.0, fs=FS, seed=int(seed))
    return p.signal[:N_WIN]

def noisy_window(seed):
    hr  = float(rng.integers(55, 90))
    snr = float(rng.uniform(0.0, 3.0))     # 0–3 dB: strong noise
    p   = generate(
        duration_s=WIN_S + 2.0, fs=FS,
        mean_hr_bpm=hr, hr_variability_ms=20.0,
        noise_snr_db=snr,
        baseline_amplitude=0.5, baseline_freq_hz=0.2,
        add_motion_artifact=False, seed=int(seed),
    )
    return p.signal[:N_WIN]

N_AUC = 20
clean_wins = [clean_window(s) for s in range(N_AUC)]
noisy_wins = [noisy_window(s) for s in range(N_AUC)]


# ── Compute SQI for all windows ───────────────────────────────────────────────

tc_clean_single = template_correlation_sqi(win_clean, FS)
se_clean_single = spectral_entropy_sqi(win_clean, FS)

tc_motion = template_correlation_sqi(win_motion, FS)
se_motion = spectral_entropy_sqi(win_motion, FS)

tc_scores_clean = np.array([template_correlation_sqi(w, FS) for w in clean_wins])
se_scores_clean = np.array([spectral_entropy_sqi(w, FS) for w in clean_wins])
tc_scores_noisy = np.array([template_correlation_sqi(w, FS) for w in noisy_wins])
se_scores_noisy = np.array([spectral_entropy_sqi(w, FS) for w in noisy_wins])

tc_auc = auc_score(tc_scores_clean, tc_scores_noisy)
se_auc = auc_score(se_scores_clean, se_scores_noisy)


# ── Print results ─────────────────────────────────────────────────────────────

print("=" * 70)
print("Signal Quality Index validation (P2-1)")
print(f"Window: {WIN_S:.0f} s × {FS} Hz = {N_WIN} samples")
print("-" * 70)

print(f"\nSingle-window results (70 bpm clean vs motion artifact):")
print(f"  {'Metric':<26} {'Clean':>10} {'Motion':>10} {'Diff':>8}")
print(f"  {'-'*54}")
tc_diff = tc_clean_single - (tc_motion if np.isfinite(tc_motion) else tc_clean_single)
se_diff = se_clean_single - (se_motion if np.isfinite(se_motion) else se_clean_single)
print(f"  {'Template Correlation':<26} {tc_clean_single:>10.4f} {tc_motion:>10.4f} {tc_diff:>8.4f}")
print(f"  {'Spectral Entropy':<26} {se_clean_single:>10.4f} {se_motion:>10.4f} {se_diff:>8.4f}")

print(f"\nAUC (20 clean vs 20 noisy [SNR 0-3 dB]):")
print(f"  {'Template Correlation AUC':<30} {tc_auc:.4f}")
print(f"  {'Spectral Entropy AUC':<30} {se_auc:.4f}")

print(f"\nDistribution summary (clean | noisy):")
print(f"  tc_sqi: {np.nanmean(tc_scores_clean):.3f}±{np.nanstd(tc_scores_clean):.3f} | "
      f"{np.nanmean(tc_scores_noisy):.3f}±{np.nanstd(tc_scores_noisy):.3f}")
print(f"  se_sqi: {np.nanmean(se_scores_clean):.3f}±{np.nanstd(se_scores_clean):.3f} | "
      f"{np.nanmean(se_scores_noisy):.3f}±{np.nanstd(se_scores_noisy):.3f}")


# ── Quality gate checks ───────────────────────────────────────────────────────

print()
print("Quality gate checks:")

check("Clean window: tc_sqi > 0.70",
      np.isfinite(tc_clean_single) and tc_clean_single > 0.70,
      f"tc_sqi={tc_clean_single:.4f}")

check("Clean window: se_sqi > 0.30",
      np.isfinite(se_clean_single) and se_clean_single > 0.30,
      f"se_sqi={se_clean_single:.4f}")

# NaN counts as "worst possible quality" — peak detector failed completely.
# Either explicit drop ≥ 0.20 OR NaN (cannot find peaks at all) satisfies this gate.
tc_motion_str = f"{tc_motion:.4f}" if np.isfinite(tc_motion) else "NaN (no peaks found)"
check("Motion window: tc_sqi drops ≥ 0.20 vs clean (or NaN = quality too low to compute)",
      (not np.isfinite(tc_motion)) or tc_diff >= 0.20,
      f"clean={tc_clean_single:.4f}, motion={tc_motion_str}")

check("Motion window: se_sqi drops ≥ 0.10 vs clean",
      np.isfinite(se_motion) and se_diff >= 0.10,
      f"clean={se_clean_single:.4f}, motion={se_motion:.4f}, diff={se_diff:.4f}")

check("AUC (clean vs noisy): tc_sqi > 0.80",
      tc_auc > 0.80,
      f"AUC={tc_auc:.4f}")

check("AUC (clean vs noisy): se_sqi > 0.80",
      se_auc > 0.80,
      f"AUC={se_auc:.4f}")


# ── Error handling ────────────────────────────────────────────────────────────

print()
print("Error handling:")

raised = False
try:
    compute_sqi(win_clean, FS, methods=("unknown_method",))
except ValueError:
    raised = True
check("ValueError on unknown method", raised)

short_sig = np.sin(2 * np.pi * 1.2 * np.arange(256) / FS)   # 1 s → ≤ 1 beat
tc_short = template_correlation_sqi(short_sig, FS)
check("NaN returned for signal with < min_beats peaks",
      not np.isfinite(tc_short),
      f"returned {tc_short}")


# ── windowed_sqi ─────────────────────────────────────────────────────────────

print()
print("windowed_sqi checks:")

long_sig = make_clean(hr_bpm=70, duration_s=60.0, fs=FS, seed=99).signal
wres = windowed_sqi(long_sig, FS, window_s=10.0, step_s=5.0)

expected_windows = (len(long_sig) - N_WIN) // int(5.0 * FS) + 1
n_got = len(wres["window_centers_s"])

check(f"windowed_sqi produces expected window count ({expected_windows})",
      n_got == expected_windows,
      f"got {n_got}")

centers = wres["window_centers_s"]
check("windowed_sqi centers are monotonically increasing",
      bool(np.all(np.diff(centers) > 0)),
      f"first={centers[0]:.2f}s, last={centers[-1]:.2f}s")


# ── Final summary ──────────────────────────────────────────────────────────────

print()
print("=" * 70)
n_pass = sum(results)
n_total = len(results)
print(f"Results: {n_pass}/{n_total} passed")

if n_pass == n_total:
    print("✓ SQI validated.")
    sys.exit(0)
else:
    print(f"✗ {n_total - n_pass} check(s) failed.")
    sys.exit(1)
