"""
Write the BIDMC end-to-end comparison to results/ (public-data results only).

Outputs:
    results/bidmc_summary.csv     median / quartile metrics per method combination
    results/bidmc_per_record.csv  per-record metrics for every combination

Run with:
    .venv/bin/python scripts/bidmc_report.py
"""

import csv
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.comparison.runner import aggregate_results, best_combination, run_comparison
from src.validation.public_dataset import load_all_bidmc

DATA_DIR = ROOT / "data" / "bidmc"
OUT_DIR = ROOT / "results"


def main() -> int:
    records = load_all_bidmc(DATA_DIR, detect_peaks=True)
    if not records:
        print(f"No BIDMC records in {DATA_DIR}. Run scripts/download_bidmc.py first.")
        return 1
    results = run_comparison(records)
    agg = aggregate_results(results)
    best = best_combination(agg)
    OUT_DIR.mkdir(exist_ok=True)

    with open(OUT_DIR / "bidmc_per_record.csv", "w", newline="") as f:
        rows = [asdict(r) for r in results]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fields = ["peak_method", "clean_method", "n_records", "n_failed",
              "hr_mae_median", "hr_mae_p75", "rmssd_err_median", "rmssd_err_p75",
              "f1_median", "f1_p25", "best"]
    with open(OUT_DIR / "bidmc_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (peak, clean), m in sorted(agg.items(),
                                       key=lambda kv: (kv[1]["hr_mae_median"],
                                                       kv[1]["rmssd_err_median"])):
            w.writerow({"peak_method": peak, "clean_method": clean,
                        **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()},
                        "best": (peak, clean) == best})
    print(f"{len(records)} records, {len(results)} evaluations; best = {best}")
    print(f"Wrote {OUT_DIR / 'bidmc_summary.csv'} and {OUT_DIR / 'bidmc_per_record.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
