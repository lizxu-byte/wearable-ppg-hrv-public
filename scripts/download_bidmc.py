"""
Download the BIDMC PPG and Respiration Dataset (53 WFDB records) into data/bidmc.

Source: https://physionet.org/content/bidmc/1.0.0/
License: Open Data Commons Attribution License v1.0 — cite Pimentel et al. (2017)
and PhysioNet (Goldberger et al., 2000) when using the data.

Run with:
    .venv/bin/python scripts/download_bidmc.py
"""

import sys
from pathlib import Path

import wfdb

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "bidmc"
RECORDS = [f"bidmc{i:02d}" for i in range(1, 54)]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    missing = [r for r in RECORDS
               if not ((DEST / f"{r}.hea").exists() and (DEST / f"{r}.dat").exists())]
    if not missing:
        print(f"All {len(RECORDS)} records already present in {DEST}")
        return 0
    print(f"Downloading {len(missing)} records from PhysioNet into {DEST} ...")
    wfdb.dl_database("bidmc", dl_dir=str(DEST), records=missing, annotators=None)
    still_missing = [r for r in RECORDS if not (DEST / f"{r}.hea").exists()]
    if still_missing:
        print(f"Missing after download: {still_missing}")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
