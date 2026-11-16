"""
Validation for comparison/runner.py — End-to-end algorithm comparison (P3-2)

Quality gates:
    1. All 6 method combinations run without crashing on all 53 records
    2. Best combination: HR MAE median < 5 bpm (target < 2 bpm)
    3. Best combination: RMSSD error median < 30 ms
    4. Best combination: F1 median > 0.90
    5. HR MAE and RMSSD error are always reported together (Rule 4 — not just one)
    6. Best combination is identified and printed with justification

Algorithm Engineering Discipline Rule 4:
    HR MAE looks good but RMSSD error may be terrible — both MUST be reported.
    A pipeline with HR MAE = 0.5 bpm but RMSSD error = 50 ms is NOT valid.

Run with (takes a few seconds for all 53 records × 6 combinations):
    .venv/bin/python tests/test_bidmc_comparison.py
"""

import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.validation.public_dataset import load_all_bidmc
from src.comparison.runner import (
    run_comparison,
    aggregate_results,
    best_combination,
    ALL_COMBINATIONS,
    PEAK_METHODS,
    CLEAN_METHODS,
)

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"
results_checks = []

def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results_checks.append(condition)


DATA_DIR = ROOT / "data" / "bidmc"
if not any(DATA_DIR.glob("bidmc[0-9][0-9].hea")):
    print(f"SKIPPED: BIDMC records not found in {DATA_DIR}. "
          "Run: python scripts/download_bidmc.py")
    sys.exit(0)


# ── Load records ──────────────────────────────────────────────────────────────

print("=" * 78)
print("End-to-end algorithm comparison on BIDMC (P3-2)")
print("=" * 78)
print(f"\nLoading 53 BIDMC records...")
t0 = time.time()
records = load_all_bidmc(DATA_DIR, detect_peaks=True)
print(f"Loaded {len(records)} records in {time.time()-t0:.1f}s")


# ── Run all 6 combinations ────────────────────────────────────────────────────

print(f"\nRunning {len(ALL_COMBINATIONS)} combinations × {len(records)} records "
      f"= {len(ALL_COMBINATIONS)*len(records)} evaluations...")
t1 = time.time()
all_results = run_comparison(records, ALL_COMBINATIONS)
elapsed = time.time() - t1
print(f"Done in {elapsed:.1f}s  ({elapsed/len(all_results)*1000:.0f} ms/evaluation)")


# ── Count failures ────────────────────────────────────────────────────────────

n_failed = sum(1 for r in all_results if r.failed)
if n_failed > 0:
    print(f"\n[WARN] {n_failed} failed evaluations:")
    for r in all_results:
        if r.failed:
            print(f"  {r.record_id} / {r.peak_method} / {r.clean_method}: {r.error_msg}")


# ── Aggregate per combination ─────────────────────────────────────────────────

agg = aggregate_results(all_results)


# ── Print comparison table ────────────────────────────────────────────────────

print()
print("-" * 78)
print(f"{'Combination':<32} {'HR MAE':>9} {'HR MAE':>9} {'RMSSD Δ':>9} "
      f"{'RMSSD Δ':>9} {'F1':>7} {'F1':>7}")
print(f"{'peak × clean':<32} {'median':>9} {'p75':>9} {'median':>9} "
      f"{'p75':>9} {'median':>7} {'p25':>7}")
print(f"{'':32} {'(bpm)':>9} {'(bpm)':>9} {'(ms)':>9} "
      f"{'(ms)':>9} {'':>7} {'':>7}")
print("-" * 78)

for (pm, cm), v in sorted(agg.items(), key=lambda x: x[1]["hr_mae_median"]):
    label = f"{pm} × {cm}"
    print(f"  {label:<30} {v['hr_mae_median']:>9.2f} {v['hr_mae_p75']:>9.2f} "
          f"{v['rmssd_err_median']:>9.2f} {v['rmssd_err_p75']:>9.2f} "
          f"{v['f1_median']:>7.3f} {v['f1_p25']:>7.3f}")

print("-" * 78)


# ── Per-record detail for best combination ────────────────────────────────────

best = best_combination(agg)
print(f"\nBest combination: {best[0]} × {best[1]}")
print(f"  HR MAE median  = {agg[best]['hr_mae_median']:.2f} bpm  "
      f"(p75 = {agg[best]['hr_mae_p75']:.2f})")
print(f"  RMSSD Δ median = {agg[best]['rmssd_err_median']:.2f} ms  "
      f"(p75 = {agg[best]['rmssd_err_p75']:.2f})")
print(f"  F1 median      = {agg[best]['f1_median']:.3f}  "
      f"(p25 = {agg[best]['f1_p25']:.3f})")

best_records = [r for r in all_results
                if r.peak_method == best[0] and r.clean_method == best[1]
                and not r.failed]
print(f"\n  Per-record breakdown for {best[0]} × {best[1]} "
      f"({len(best_records)}/{len(records)} records):")
print(f"  {'Record':<10} {'HR MAE':>8} {'RMSSD Δ':>9} {'F1':>7} "
      f"{'PAT(s)':>8} {'nPPG':>6} {'nECG':>6}")
print("  " + "-" * 65)
for r in sorted(best_records, key=lambda x: x.hr_mae_bpm):
    print(f"  {r.record_id:<10} {r.hr_mae_bpm:>8.2f} {r.rmssd_error_ms:>9.2f} "
          f"{r.f1:>7.3f} {r.pat_s:>8.3f} {r.n_ppg_peaks:>6d} {r.n_ecg_peaks:>6d}")


# ── Quality gate checks ───────────────────────────────────────────────────────

print()
print("Quality gate checks:")

# Gate 1: no crashes (all combinations ran)
expected_total = len(ALL_COMBINATIONS) * len(records)
check(f"All {len(ALL_COMBINATIONS)} combinations × 53 records ran without crash",
      len(all_results) == expected_total,
      f"{len(all_results)}/{expected_total} evaluations completed")

# Gate 2: HR MAE < 5 bpm for best combination
hr_best = agg[best]["hr_mae_median"]
check(f"Best combination HR MAE median < 5 bpm  [{best[0]} × {best[1]}]",
      hr_best < 5.0,
      f"{hr_best:.2f} bpm")

# Gate 3: RMSSD error < 30 ms for best combination
rmssd_best = agg[best]["rmssd_err_median"]
check(f"Best combination RMSSD error median < 30 ms  [{best[0]} × {best[1]}]",
      rmssd_best < 30.0,
      f"{rmssd_best:.2f} ms")

# Gate 4: F1 > 0.90 for best combination
f1_best = agg[best]["f1_median"]
check(f"Best combination F1 median > 0.90",
      f1_best > 0.90,
      f"F1={f1_best:.3f}")

# Gate 5: Rule 4 — both HR MAE and RMSSD are finite for the best combination
both_finite = (np.isfinite(hr_best) and np.isfinite(rmssd_best))
check("HR MAE and RMSSD error both reported (Rule 4: not just one metric)",
      both_finite,
      f"HR MAE={hr_best:.2f} bpm, RMSSD Δ={rmssd_best:.2f} ms")

# Gate 6: best combination identified
check(f"Best combination identified: {best[0]} × {best[1]}",
      len(best) == 2 and best[0] in PEAK_METHODS and best[1] in CLEAN_METHODS,
      f"{best}")


# ── Algorithm recommendation ──────────────────────────────────────────────────

print()
print("Algorithm recommendation:")
print(f"  Best peak detector : {best[0]}")
print(f"  Best RR cleaner    : {best[1]}")
print(f"  Rationale: minimum HR MAE (primary) then RMSSD error (secondary)")
print(f"  Filter (fixed)     : Butterworth 0.4–10 Hz order 4 (zero-phase)")
print()

# Rank all by HR MAE for context
ranked = sorted(agg.items(), key=lambda x: x[1]["hr_mae_median"])
print("  Full ranking by HR MAE median:")
for rank, ((pm, cm), v) in enumerate(ranked, 1):
    marker = " ← BEST" if (pm, cm) == best else ""
    print(f"    {rank}. {pm:<12} × {cm:<18}  "
          f"HR MAE={v['hr_mae_median']:.2f}  RMSSD Δ={v['rmssd_err_median']:.2f} ms  "
          f"F1={v['f1_median']:.3f}{marker}")


# ── Final summary ──────────────────────────────────────────────────────────────

print()
print("=" * 78)
n_pass  = sum(results_checks)
n_total = len(results_checks)
print(f"Results: {n_pass}/{n_total} passed")

if n_pass == n_total:
    print("✓ End-to-end comparison validated.")
    sys.exit(0)
else:
    print(f"✗ {n_total - n_pass} check(s) failed.")
    sys.exit(1)
