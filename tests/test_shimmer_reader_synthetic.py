"""
Validation for io/shimmer_reader.py on synthetic Shimmer-format CSV files.

Participant recordings are not distributed with this repository, so this test
writes Shimmer ConsensysPRO-style exports from a synthetic PPG into a temporary
directory and checks that the reader recovers them.

Checks:
    1. Format A (sep= line, header, unit row): PPG column, fs, sample count
    2. Format A: PPG values round-trip within write precision
    3. Format A: accelerometer (N, 3) and GSR channels are loaded
    4. Format B (sep= line, no unit row) parses to the same PPG
    5. Format C (plain CSV, unit row) parses to the same PPG
    6. Truncated timestamps (e.g. 1.77E+12) fall back to nominal fs
    7. Missing PPG column raises ValueError

Run with:
    .venv/bin/python tests/test_shimmer_reader_synthetic.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from src.io.shimmer_reader import load_session
from src.validation.synthetic import generate

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"
results = []


def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)


cfg = yaml.safe_load(open(ROOT / "config.yaml"))
FS = int(cfg["sampling"]["fs"])
D = cfg["data"]

PPG_COL = D["ppg_column_priority"][0]
TS_COL = D["timestamp_column"]
ACC_COLS = D["accel_columns"]["ln"] + D["accel_columns"]["wr"]
GSR_COLS = [D["gsr_columns"]["conductance"], D["gsr_columns"]["resistance"]]
HEADERS = [TS_COL, PPG_COL] + ACC_COLS + GSR_COLS
UNITS = ["ms", "mV"] + ["m/(s^2)"] * 6 + ["uS", "kOhms"]

sig = generate(duration_s=60.0, fs=FS, mean_hr_bpm=72.0, seed=42)
N = len(sig.signal)
ppg_mv = 1500.0 + 200.0 * sig.signal          # Shimmer PPG_A13 is in the 1–2 V range
ts_ms = 1.7e12 + np.arange(N) * 1000.0 / FS
rng = np.random.default_rng(42)
acc = np.column_stack([rng.normal(0, 0.05, N) + g for g in (0.0, 0.0, 9.81)] * 2)
gsr_us = 2.0 + 0.01 * np.arange(N) / FS
gsr_kohm = 1000.0 / gsr_us


def write_csv(path: Path, sep: str, sep_line: bool, unit_row: bool,
              ts=None) -> None:
    ts = ts_ms if ts is None else ts
    lines = []
    if sep_line:
        lines.append("sep=\\t" if sep == "\t" else f"sep={sep}")
    lines.append(sep.join(HEADERS))
    if unit_row:
        lines.append(sep.join(UNITS))
    for i in range(N):
        row = [f"{ts[i]:.3f}" if isinstance(ts[i], float) else str(ts[i]),
               f"{ppg_mv[i]:.6f}",
               *(f"{v:.6f}" for v in acc[i]),
               f"{gsr_us[i]:.6f}", f"{gsr_kohm[i]:.6f}"]
        lines.append(sep.join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


print("=" * 70)
print("Shimmer reader validation on synthetic exports")
print("-" * 70)

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    # Format A
    pa = tmp / "synthetic_formatA.csv"
    write_csv(pa, "\t", sep_line=True, unit_row=True)
    sa = load_session(pa, cfg)
    check("Format A: PPG column resolved", sa.ppg_column_used == PPG_COL, sa.ppg_column_used)
    check("Format A: fs within 256 ± 1 Hz", abs(sa.fs_measured - FS) <= 1.0,
          f"fs={sa.fs_measured:.2f} Hz, source={sa.metadata['fs_source']}")
    check("Format A: sample count preserved", sa.n_samples == N, f"{sa.n_samples} vs {N}")
    max_err = float(np.max(np.abs(sa.ppg - ppg_mv)))
    check("Format A: PPG round-trip error < 1e-5 mV", max_err < 1e-5, f"max err={max_err:.2e}")
    check("Format A: accel arrays (N, 3) and finite",
          sa.accel_ln.shape == (N, 3) and sa.accel_wr.shape == (N, 3)
          and np.isfinite(sa.accel_wr).all(),
          f"ln={sa.accel_ln.shape}, wr={sa.accel_wr.shape}")
    check("Format A: GSR conductance loaded",
          np.allclose(sa.gsr_conductance, gsr_us, atol=1e-5))

    # Format B
    pb = tmp / "synthetic_formatB.csv"
    write_csv(pb, "\t", sep_line=True, unit_row=False)
    sb = load_session(pb, cfg)
    check("Format B (no unit row): same PPG", sb.n_samples == N and np.allclose(sb.ppg, ppg_mv, atol=1e-5))

    # Format C
    pc = tmp / "synthetic_formatC.csv"
    write_csv(pc, ",", sep_line=False, unit_row=True)
    sc = load_session(pc, cfg)
    check("Format C (plain CSV + unit row): same PPG",
          sc.n_samples == N and np.allclose(sc.ppg, ppg_mv, atol=1e-5))

    # Truncated timestamps
    pt = tmp / "synthetic_truncated_ts.csv"
    write_csv(pt, "\t", sep_line=True, unit_row=True, ts=["1.77E+12"] * N)
    st = load_session(pt, cfg)
    check("Truncated timestamps fall back to nominal fs",
          st.fs_measured == float(FS) and st.metadata["fs_source"] == "fallback_nominal"
          and abs(st.duration_s - (N - 1) / FS) < 1e-9,
          f"fs={st.fs_measured}, source={st.metadata['fs_source']}")

    # Missing PPG column
    pm = tmp / "synthetic_no_ppg.csv"
    pm.write_text("sep=\\t\n" + "\t".join([TS_COL, "Other_CAL"]) + "\nms\tmV\n"
                  + "".join(f"{ts_ms[i]:.3f}\t1.0\n" for i in range(20)), encoding="utf-8")
    try:
        load_session(pm, cfg)
        raised = False
    except ValueError:
        raised = True
    check("Missing PPG column raises ValueError", raised)

n_pass, n_total = sum(results), len(results)
print("-" * 70)
print(f"Results: {n_pass}/{n_total} passed")
if n_pass == n_total:
    print("✓ Shimmer reader validated on synthetic exports.")
    sys.exit(0)
print(f"✗ {n_total - n_pass} check(s) failed.")
sys.exit(1)
