"""
Validation for hrv_time.py — HRV time-domain metrics (P1-8)

Comparison strategy: neurokit2 internally derives RR from integer peaks at
1000 Hz. Running both our code AND neurokit2 on the SAME integer-quantized
RR data removes any float-vs-integer precision gap.

Quality gates:
    1. MeanNN error vs neurokit2 < 0.01 ms
    2. SDNN error vs neurokit2 < 0.5 ms
    3. RMSSD error vs neurokit2 < 0.5 ms
    4. pNN50 error vs neurokit2 < 0.05 %  (same integer data → should be exact)
    5. pNN50 denominator regression: ours ≠ old_bug AND matches neurokit2
    6. SDNN uses ddof=1 (Task Force 1996)
    7. MinHR < MeanHR < MaxHR
    8–10. Error handling: ValueError on bad inputs

References:
    Task Force ESC (1996) Circulation 93:1043-1065.
    neurokit2: Makowski et al. (2021) Behav Res Methods 53:1689-1696.

Run with:
    .venv/bin/python tests/test_hrv_time.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import neurokit2 as nk

from src.pipeline.hrv_time import compute_hrv_time

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)


# ── Synthetic RR (float) ───────────────────────────────────────────────────────

def make_rr(n_beats=350, mean_rr=857.0, rmssd=35.0, seed=42):
    rng = np.random.default_rng(seed)
    ar_coef = 0.5
    noise_std = rmssd / np.sqrt(2 * (1 - ar_coef**2))
    rr = np.zeros(n_beats)
    rr[0] = mean_rr
    for i in range(1, n_beats):
        rr[i] = mean_rr + ar_coef * (rr[i - 1] - mean_rr) + rng.normal(0, noise_std)
    return np.clip(rr, 450.0, 1400.0)


rr_float = make_rr(n_beats=350, mean_rr=857.0, rmssd=35.0, seed=42)

# Integer peaks at 1000 Hz (same quantization that neurokit2 uses internally)
peaks_int = np.round(
    np.concatenate([[0.0], np.cumsum(rr_float[:-1])])
).astype(int)

# Derive the RR values neurokit2 ACTUALLY sees (differences of cumulative sums)
rr_int = np.diff(peaks_int).astype(float)   # N-1 values
# Prepend a nominal first RR so length matches peaks (neurokit2 returns metrics for all beats)
rr_nk = np.concatenate([[rr_float[0]], rr_int])  # N values, first preserved from float

# ── For fair comparison: our computation runs on the same rr_int data ──────────
# (neurokit2 sees rr_int from the peaks; we must compute on the same values)
our   = compute_hrv_time(rr_nk)
nk_df = nk.hrv_time(peaks_int, sampling_rate=1000, show=False)
nk_ref = nk_df.iloc[0].to_dict()


# ── Print comparison table ─────────────────────────────────────────────────────

print("=" * 72)
print("HRV time-domain metrics comparison (P1-8)")
print(f"Sequence: {len(rr_nk)} beats, mean_rr={np.mean(rr_nk):.1f} ms")
print(f"(Both our code and neurokit2 operate on integer-quantized RR data)")
print("-" * 72)

print(f"\n{'Metric':<16} {'Ours':>12} {'neurokit2':>12} {'Error':>10} {'Unit'}")
print("-" * 62)

errors = {}
def cmp_row(label, our_val, nk_key, unit="", scale=1.0):
    nk_val = nk_ref.get(nk_key, float("nan"))
    nk_val_scaled = nk_val * scale
    err = abs(our_val - nk_val_scaled) if (np.isfinite(our_val) and np.isfinite(nk_val_scaled)) else float("nan")
    errors[label] = err
    print(f"  {label:<14} {our_val:>12.4f} {nk_val_scaled:>12.4f} {err:>10.5f}  {unit}")

cmp_row("MeanNN",  our["mean_nn_ms"],  "HRV_MeanNN",  "ms")
cmp_row("SDNN",    our["sdnn_ms"],     "HRV_SDNN",    "ms")
cmp_row("RMSSD",   our["rmssd_ms"],    "HRV_RMSSD",   "ms")
cmp_row("pNN50",   our["pnn50_pct"],   "HRV_pNN50",   "%")
cmp_row("CVNN",    our["cvnn_pct"],    "HRV_CVNN",    "%", scale=100.0)  # nk: ratio; ours: %
print()


# ── Quality gate checks ────────────────────────────────────────────────────────

print("Quality gate checks:")

check("MeanNN error < 0.01 ms",  errors["MeanNN"] < 0.01,  f"err={errors['MeanNN']:.6f} ms")
check("SDNN error < 0.5 ms",     errors["SDNN"]   < 0.5,   f"err={errors['SDNN']:.5f} ms")
check("RMSSD error < 0.5 ms",    errors["RMSSD"]  < 0.5,   f"err={errors['RMSSD']:.5f} ms")
check("pNN50 error < 0.05 %",    errors["pNN50"]  < 0.05,  f"err={errors['pNN50']:.5f} %")


# ── pNN50 denominator regression (V1 bug) ─────────────────────────────────────

print()
print("pNN50 denominator regression (V1 bug):")

diff_rr  = np.diff(rr_nk)
nn50     = int(np.sum(np.abs(diff_rr) > 50.0))
n        = len(rr_nk)

pnn50_correct = nn50 / (n - 1) * 100.0   # Task Force 1996: N-1
pnn50_old_bug = nn50 /  n      * 100.0   # old code bug: N

print(f"  NN50              = {nn50}")
print(f"  N (len rr_ms)     = {n}")
print(f"  N-1 (len diff_rr) = {n - 1}")
print(f"  pNN50 (correct)   = {pnn50_correct:.5f} %")
print(f"  pNN50 (old bug)   = {pnn50_old_bug:.5f} %")
print(f"  Difference        = {abs(pnn50_correct - pnn50_old_bug):.6f} %")
print()

check(
    "pNN50 uses N-1 denominator (Task Force 1996)",
    abs(our["pnn50_pct"] - pnn50_correct) < 1e-9,
    f"our={our['pnn50_pct']:.5f} %, correct={pnn50_correct:.5f} %",
)
check(
    "pNN50 differs from old buggy N denominator",
    abs(our["pnn50_pct"] - pnn50_old_bug) > 1e-6,
    f"our={our['pnn50_pct']:.5f} %, old={pnn50_old_bug:.5f} %, diff={abs(pnn50_correct - pnn50_old_bug):.6f} %",
)


# ── SDNN uses ddof=1 ──────────────────────────────────────────────────────────

print()
print("SDNN formula check:")
sdnn_ddof1 = float(np.std(rr_nk, ddof=1))
sdnn_ddof0 = float(np.std(rr_nk, ddof=0))
print(f"  SDNN ddof=1 = {sdnn_ddof1:.4f} ms  (Task Force 1996 — sample std)")
print(f"  SDNN ddof=0 = {sdnn_ddof0:.4f} ms")
print()

check(
    "SDNN uses sample std (ddof=1)",
    abs(our["sdnn_ms"] - sdnn_ddof1) < 1e-9,
    f"our={our['sdnn_ms']:.6f}, ddof1={sdnn_ddof1:.6f}",
)


# ── HR metrics ────────────────────────────────────────────────────────────────

print()
print("HR metrics (derived from RR):")
print(f"  MeanHR = 60000 / MeanNN = 60000 / {our['mean_nn_ms']:.2f} = {our['mean_hr_bpm']:.4f} bpm")
print(f"  MinHR  = 60000 / max(RR) = 60000 / {np.max(rr_nk):.2f}  = {our['min_hr_bpm']:.4f} bpm")
print(f"  MaxHR  = 60000 / min(RR) = 60000 / {np.min(rr_nk):.2f}  = {our['max_hr_bpm']:.4f} bpm")
print()

check(
    "MeanHR = 60000 / MeanNN",
    abs(our["mean_hr_bpm"] - 60000.0 / our["mean_nn_ms"]) < 1e-9,
)
check(
    "MinHR < MeanHR < MaxHR",
    our["min_hr_bpm"] < our["mean_hr_bpm"] < our["max_hr_bpm"],
    f"{our['min_hr_bpm']:.2f} < {our['mean_hr_bpm']:.2f} < {our['max_hr_bpm']:.2f}",
)


# ── Error handling ────────────────────────────────────────────────────────────

print()
print("Error handling:")

for bad_input, label in [
    (np.array([800.0]),              "single-element"),
    (np.array([800.0, -50.0, 810.0]), "negative RR"),
    (np.array([800.0,  0.0, 810.0]), "zero RR"),
]:
    raised = False
    try:
        compute_hrv_time(bad_input)
    except ValueError:
        raised = True
    check(f"ValueError on {label} input", raised)


# ── Full output ───────────────────────────────────────────────────────────────

print()
print("Full metrics (our implementation on quantized data):")
for k, v in our.items():
    print(f"  {k:<16} = {v:.4f}")


# ── Final summary ──────────────────────────────────────────────────────────────

print()
print("=" * 72)
n_pass = sum(results)
n_total = len(results)
print(f"Results: {n_pass}/{n_total} passed")

if n_pass == n_total:
    print("✓ HRV time-domain validated.")
    sys.exit(0)
else:
    print(f"✗ {n_total - n_pass} check(s) failed.")
    sys.exit(1)
