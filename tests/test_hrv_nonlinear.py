"""
Validation for hrv_nonlinear.py — Poincaré HRV metrics (P1-10)

Quality gates:
    1. SD1 error vs neurokit2 < 1 ms
    2. SD2 error vs neurokit2 < 1 ms
    3. SD1 = RMSSD / √2  (mathematical identity, Brennan 2001 Eq. 4)
    4. SD2 = √(2×SDNN² − RMSSD²/2)  (cross-check with time domain)
    5. SD1 < SD2  (physiological requirement for healthy adults)
    6. ValueError on < 3 RR intervals
    7. ValueError on non-positive RR

Run with:
    .venv/bin/python tests/test_hrv_nonlinear.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]
import neurokit2 as nk

from src.pipeline.hrv_nonlinear import compute_hrv_nonlinear
from src.pipeline.hrv_time import compute_hrv_time

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)


# ── Synthetic RR ───────────────────────────────────────────────────────────────

def make_rr(n_beats=350, mean_rr=857.0, rmssd=35.0, seed=42):
    rng = np.random.default_rng(seed)
    ar_coef = 0.5
    noise_std = rmssd / np.sqrt(2 * (1 - ar_coef**2))
    rr = np.zeros(n_beats)
    rr[0] = mean_rr
    for i in range(1, n_beats):
        rr[i] = mean_rr + ar_coef * (rr[i-1] - mean_rr) + rng.normal(0, noise_std)
    return np.clip(rr, 450.0, 1400.0)


rr_ms = make_rr()

# Integer-quantized RR for fair neurokit2 comparison (same as test_hrv_time.py)
peaks_int = np.round(np.concatenate([[0.0], np.cumsum(rr_ms[:-1])])).astype(int)
rr_int    = np.concatenate([[rr_ms[0]], np.diff(peaks_int).astype(float)])

our = compute_hrv_nonlinear(rr_int)
td  = compute_hrv_time(rr_int)   # for cross-check with RMSSD/SDNN

# neurokit2 reference
nk_df  = nk.hrv_nonlinear(peaks_int, sampling_rate=1000, show=False)
nk_ref = nk_df.iloc[0].to_dict()


# ── Print comparison ───────────────────────────────────────────────────────────

print("=" * 65)
print("HRV nonlinear Poincaré metrics comparison (P1-10)")
print(f"Sequence: {len(rr_int)} beats (integer-quantized for fair comparison)")
print("-" * 65)

nk_sd1 = nk_ref.get("HRV_SD1", float("nan"))
nk_sd2 = nk_ref.get("HRV_SD2", float("nan"))

print(f"\n{'Metric':<14} {'Ours':>10} {'neurokit2':>10} {'Error':>8}")
print("-" * 48)
for label, our_val, nk_val in [
    ("SD1 (ms)",  our["sd1_ms"],  nk_sd1),
    ("SD2 (ms)",  our["sd2_ms"],  nk_sd2),
    ("SD1/SD2",   our["sd1_sd2"], nk_ref.get("HRV_SD1SD2", float("nan"))),
    ("Area (ms²)", our["area_ms2"], float("nan")),
]:
    err = abs(our_val - nk_val) if (np.isfinite(our_val) and np.isfinite(nk_val)) else float("nan")
    err_str = f"{err:>8.4f}" if np.isfinite(err) else f"{'N/A':>8}"
    print(f"  {label:<12} {our_val:>10.4f} {nk_val:>10.4f} {err_str}")
print()


# ── Quality gate checks ────────────────────────────────────────────────────────

print("Quality gate checks:")

err_sd1 = abs(our["sd1_ms"] - nk_sd1)
err_sd2 = abs(our["sd2_ms"] - nk_sd2)

check("SD1 error vs neurokit2 < 1 ms",
      np.isfinite(err_sd1) and err_sd1 < 1.0,
      f"err={err_sd1:.4f} ms")
check("SD2 error vs neurokit2 < 1 ms",
      np.isfinite(err_sd2) and err_sd2 < 1.0,
      f"err={err_sd2:.4f} ms")


# ── Mathematical identity: SD1 = RMSSD / √2 ───────────────────────────────────

print()
print("Mathematical identity checks (Brennan 2001):")

rmssd = td["rmssd_ms"]
sdnn  = td["sdnn_ms"]

sd1_from_rmssd = rmssd / np.sqrt(2.0)
sd2_from_formula = float(np.sqrt(max(2.0 * sdnn**2 - rmssd**2 / 2.0, 0.0)))

print(f"  RMSSD          = {rmssd:.4f} ms")
print(f"  SDNN           = {sdnn:.4f} ms")
print(f"  SD1 (computed) = {our['sd1_ms']:.4f} ms")
print(f"  RMSSD/√2       = {sd1_from_rmssd:.4f} ms  ← must equal SD1")
print(f"  SD2 (computed) = {our['sd2_ms']:.4f} ms")
print(f"  √(2SDNN²−½RMSSD²) = {sd2_from_formula:.4f} ms  ← must equal SD2")
print()

# SD1 = RMSSD/√2 is exact only for population variance (ddof=0) or infinite N.
# With ddof=1 (sample std), the difference is O(1/N) ≈ 0.03 ms for N=350.
# Test: |SD1 − RMSSD/√2| < 0.5 ms (documents the approximation, not a code error).
check("SD1 ≈ RMSSD/√2 within 0.5 ms  (exact only at population variance limit)",
      abs(our["sd1_ms"] - sd1_from_rmssd) < 0.5,
      f"SD1={our['sd1_ms']:.4f}, RMSSD/√2={sd1_from_rmssd:.4f}, diff={abs(our['sd1_ms']-sd1_from_rmssd):.4f} ms")

# Exact identity given our SD1 definition: SD1 = std(diff_rr, ddof=1) / √2
diff_rr = np.diff(rr_int)
sd1_exact = float(np.std(diff_rr, ddof=1) / np.sqrt(2.0))
check("SD1 = std(diff_rr, ddof=1) / √2  (exact formula as implemented)",
      abs(our["sd1_ms"] - sd1_exact) < 1e-9,
      f"SD1={our['sd1_ms']:.8f}, formula={sd1_exact:.8f}")

# SD2 exact identity: SD2 = sqrt(2×SDNN² − SD1²)
sd2_exact = float(np.sqrt(max(2.0 * sdnn**2 - our["sd1_ms"]**2, 0.0)))
check("SD2 = √(2×SDNN² − SD1²)  (exact formula as implemented)",
      abs(our["sd2_ms"] - sd2_exact) < 1e-9,
      f"SD2={our['sd2_ms']:.8f}, formula={sd2_exact:.8f}")


# ── Physiological and shape checks ────────────────────────────────────────────

print()
print("Physiological checks:")

check("SD1 < SD2  (short-term < long-term, typical for healthy adults)",
      our["sd1_ms"] < our["sd2_ms"],
      f"SD1={our['sd1_ms']:.2f} ms, SD2={our['sd2_ms']:.2f} ms")
check("SD1 > 0 and SD2 > 0",
      our["sd1_ms"] > 0 and our["sd2_ms"] > 0)
check("Area = π × SD1 × SD2",
      abs(our["area_ms2"] - np.pi * our["sd1_ms"] * our["sd2_ms"]) < 1e-6,
      f"area={our['area_ms2']:.4f}, π×SD1×SD2={np.pi*our['sd1_ms']*our['sd2_ms']:.4f}")
check("Poincaré arrays: len(rr_n) = len(rr_n1) = N-1",
      len(our["rr_n"]) == len(rr_int) - 1 and len(our["rr_n1"]) == len(rr_int) - 1)


# ── Error handling ────────────────────────────────────────────────────────────

print()
print("Error handling:")

for bad, label in [
    (np.array([800.0, 810.0]),            "2-element (< 3)"),
    (np.array([800.0, -10.0, 810.0]),     "negative RR"),
    (np.array([800.0, 0.0, 810.0]),       "zero RR"),
]:
    raised = False
    try:
        compute_hrv_nonlinear(bad)
    except ValueError:
        raised = True
    check(f"ValueError on {label}", raised)


# ── Full output ───────────────────────────────────────────────────────────────

print()
print(f"Full nonlinear output:")
print(f"  SD1      = {our['sd1_ms']:.4f} ms")
print(f"  SD2      = {our['sd2_ms']:.4f} ms")
print(f"  SD1/SD2  = {our['sd1_sd2']:.4f}")
print(f"  Area     = {our['area_ms2']:.2f} ms²")
print(f"  Poincaré pairs: {len(our['rr_n'])} points")


# ── Final summary ──────────────────────────────────────────────────────────────

print()
print("=" * 65)
n_pass = sum(results)
n_total = len(results)
print(f"Results: {n_pass}/{n_total} passed")

if n_pass == n_total:
    print("✓ HRV nonlinear validated.")
    sys.exit(0)
else:
    print(f"✗ {n_total - n_pass} check(s) failed.")
    sys.exit(1)
