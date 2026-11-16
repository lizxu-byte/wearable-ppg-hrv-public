"""
shimmer_reader.py — Shimmer3 GSR+ CSV parser

Handles all known Shimmer ConsensysPRO export variants:
  Format A (standard): row0=sep=\\t, row1=column names, row2=units, row3+=data
  Format B (no unit row): row0=sep=\\t, row1=column names, row2+=data
  Format C (plain CSV/TSV): row0=column names, row1=units or data

All column names and priority lists come from config.yaml — no hardcoded strings
in algorithm logic.

Device: Shimmer3 GSR+ with optical PPG probe
Nominal sampling rate: 256 Hz (measured from timestamp deltas per session)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# ── Config loading ─────────────────────────────────────────────────────────────

def _load_config(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(config: Optional[dict] = None) -> Path:
    """
    Directory holding the local Shimmer subject folders (0/ … 66/).

    SHIMMER_DATA_ROOT overrides config.yaml data.root_dir; a relative path is
    resolved against the repository root. Participant recordings are never part
    of the repository, so callers must handle a missing directory.
    """
    if config is None:
        config = _load_config()
    root = Path(os.environ.get("SHIMMER_DATA_ROOT", config["data"]["root_dir"])).expanduser()
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    return root


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class SessionData:
    """
    Parsed output of one Shimmer session CSV.

    All numpy arrays are 1-D with length N (number of samples) except
    accel_ln and accel_wr which are (N, 3).

    Units:
        ppg              : mV
        timestamps_ms    : Unix milliseconds
        time_s           : seconds from session start (t=0)
        accel_ln/wr      : m/s²
        gsr_conductance  : µS
        gsr_resistance   : kΩ
        fs_measured      : Hz (computed from median inter-sample interval)
    """
    ppg: np.ndarray
    timestamps_ms: np.ndarray
    time_s: np.ndarray
    accel_ln: np.ndarray              # shape (N, 3)
    accel_wr: np.ndarray              # shape (N, 3)
    gsr_conductance: np.ndarray       # µS
    gsr_resistance: np.ndarray        # kΩ
    fs_measured: float
    fs_nominal: int
    ppg_column_used: str
    source_path: str
    n_samples: int
    duration_s: float
    metadata: dict = field(default_factory=dict)


# ── Separator / header detection ───────────────────────────────────────────────

_UNIT_KEYWORDS = frozenset([
    "ms", "msec", "msecs", "milliseconds",
    "mv", "millivolts", "v", "volts",
    "kohms", "kohm", "ohms", "ohm",
    "kpa", "pa", "degrees", "celsius",
    "no_units", "no units", "m/(s^2)",
    "local_flux", "us", "a.u.", "au",
])


def _detect_sep_from_first_line(line: str) -> Optional[str]:
    """Return separator character if line is a Shimmer 'sep=' header, else None."""
    clean = line.replace("\r", "").replace("\n", "").replace('"', "").replace("'", "").lower()
    if not clean.startswith("sep="):
        return None
    val = clean[len("sep="):]
    if val in ("\t", "\\t", ""):
        return "\t"
    if val == ",":
        return ","
    if val == ";":
        return ";"
    return "\t"  # Shimmer default


def _guess_sep(line: str) -> str:
    counts = {"\t": line.count("\t"), ",": line.count(","), ";": line.count(";")}
    return max(counts, key=counts.get)


def _line_is_unit_row(line: str) -> bool:
    tokens = set(re.split(r"[\t,;\s]+", line.lower()))
    return bool(tokens & _UNIT_KEYWORDS)


def _clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().replace("﻿", "") for c in df.columns]
    return df


# ── Core CSV reader ────────────────────────────────────────────────────────────

def safe_read_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read a Shimmer CSV into a DataFrame, handling all known export formats.

    Tries encodings in order: utf-8-sig → utf-8 → gbk → latin1.
    Detects sep= header line, identifies and skips unit row automatically.

    Parameters
    ----------
    csv_path : Path
        Absolute path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column names (BOM and whitespace stripped).
        All data rows as strings; callers are responsible for numeric conversion.

    Raises
    ------
    ValueError
        If the file cannot be read with any supported encoding.
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
    last_err: Optional[Exception] = None

    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, errors="ignore") as f:
                line1 = f.readline()
                line2 = f.readline()
                line3 = f.readline()

            sep_detected = _detect_sep_from_first_line(line1)

            if sep_detected is not None:
                # Shimmer file: line1=sep=, line2=headers, line3=units or data
                sep = sep_detected
                skip = [0, 2] if _line_is_unit_row(line3) else [0]
            else:
                # Plain CSV/TSV: line1=headers, line2=units or data
                sep = _guess_sep(line1)
                skip = [1] if _line_is_unit_row(line2) else []

            df = pd.read_csv(csv_path, sep=sep, skiprows=skip,
                             encoding=enc, low_memory=False)
            return _clean_col_names(df)

        except Exception as e:
            last_err = e
            continue

    raise ValueError(f"Cannot read {csv_path}: {last_err}")


# ── Column finders ─────────────────────────────────────────────────────────────

def find_ppg_column(df: pd.DataFrame, priority: list[str]) -> str:
    """
    Return the first column whose name (case-insensitive) contains any
    keyword from the priority list.

    Parameters
    ----------
    df : pd.DataFrame
    priority : list[str]
        Ordered list of substrings to match, e.g. ["PPG_A13_CAL", "PPG_CAL", "PPG"].

    Returns
    -------
    str  Column name.

    Raises
    ------
    ValueError  If no PPG column is found.
    """
    cols_upper = {c: c.upper() for c in df.columns}
    for kw in priority:
        kw_up = kw.upper()
        matches = [c for c, cu in cols_upper.items() if kw_up in cu]
        if matches:
            return matches[0]
    raise ValueError(
        f"No PPG column found. Tried keywords {priority}. "
        f"Available columns: {df.columns.tolist()[:20]}"
    )


def find_column(df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
    """Return first column containing any keyword (case-insensitive), or None."""
    for kw in keywords:
        kw_up = kw.upper()
        matches = [c for c in df.columns if kw_up in c.upper()]
        if matches:
            return matches[0]
    return None


# ── Sampling rate computation ──────────────────────────────────────────────────

def compute_fs(timestamps_ms: np.ndarray) -> tuple[float, str]:
    """
    Estimate sampling rate from Unix millisecond timestamps.

    Uses the median inter-sample interval to be robust against occasional
    dropped packets or timestamp jitter.

    Some early Shimmer exports truncate timestamps to 3 significant figures
    (e.g. 1.77E+12), making all differences zero. In that case the function
    returns (nan, "truncated") so callers can fall back to the nominal rate.

    Parameters
    ----------
    timestamps_ms : np.ndarray
        Unix timestamps in milliseconds, length ≥ 2.

    Returns
    -------
    tuple[float, str]
        (fs_hz, source) where source is "measured" or "truncated".
    """
    if len(timestamps_ms) < 2:
        return float("nan"), "truncated"
    diffs = np.diff(np.sort(timestamps_ms.astype(float)))
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(diffs) == 0:
        return float("nan"), "truncated"
    median_dt_ms = float(np.median(diffs))
    return 1000.0 / median_dt_ms, "measured"


# ── Top-level session loader ───────────────────────────────────────────────────

def load_session(csv_path: Path, config: Optional[dict] = None) -> SessionData:
    """
    Parse one Shimmer session CSV into a SessionData object.

    Reads all channels defined in config.yaml (PPG, accelerometers, GSR).
    Computes measured sampling rate from timestamp deltas.

    Parameters
    ----------
    csv_path : Path
        Path to a single session CSV file.
    config : dict, optional
        Parsed config.yaml dict. Loaded from disk if not provided.

    Returns
    -------
    SessionData
        All signal arrays, measured fs, and metadata.

    Raises
    ------
    ValueError
        If the CSV has too few samples or required columns are missing.
    """
    if config is None:
        config = _load_config()

    cfg_data = config["data"]
    cfg_samp = config["sampling"]

    df = safe_read_csv(csv_path)

    # ── PPG ───────────────────────────────────────────────────────────────────
    ppg_col = find_ppg_column(df, cfg_data["ppg_column_priority"])
    ppg = pd.to_numeric(df[ppg_col], errors="coerce").to_numpy(dtype=float)

    # ── Timestamp ─────────────────────────────────────────────────────────────
    ts_col = cfg_data["timestamp_column"]
    if ts_col not in df.columns:
        # fallback: any column with TIMESTAMP or TIME
        ts_col = find_column(df, ["TIMESTAMPSYNC", "TIMESTAMP", "TIME"])
        if ts_col is None:
            raise ValueError(f"No timestamp column found in {csv_path}")
    timestamps_ms = pd.to_numeric(df[ts_col], errors="coerce").to_numpy(dtype=float)

    # Drop rows where either PPG or timestamp is NaN
    valid = np.isfinite(ppg) & np.isfinite(timestamps_ms)
    ppg = ppg[valid]
    timestamps_ms = timestamps_ms[valid]

    if len(ppg) < 10:
        raise ValueError(f"Too few valid samples ({len(ppg)}) in {csv_path}")

    fs_measured, fs_source = compute_fs(timestamps_ms)
    fs_nominal = int(cfg_samp["fs"])

    if not np.isfinite(fs_measured):
        # Timestamps truncated (early Shimmer export artefact); fall back to nominal
        fs_measured = float(fs_nominal)
        fs_source = "fallback_nominal"
        # Reconstruct time axis from sample index at nominal rate
        time_s = np.arange(len(ppg)) / float(fs_nominal)
    else:
        time_s = (timestamps_ms - timestamps_ms[0]) / 1000.0

    # ── Accelerometers ────────────────────────────────────────────────────────
    def _load_accel(col_names: list[str]) -> np.ndarray:
        arrays = []
        for col in col_names:
            if col in df.columns:
                arrays.append(pd.to_numeric(df[col], errors="coerce").to_numpy(float)[valid])
            else:
                arrays.append(np.full(len(ppg), np.nan))
        return np.column_stack(arrays)  # (N, 3)

    accel_ln = _load_accel(cfg_data["accel_columns"]["ln"])
    accel_wr = _load_accel(cfg_data["accel_columns"]["wr"])

    # ── GSR ───────────────────────────────────────────────────────────────────
    def _load_col(col_name: str) -> np.ndarray:
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce").to_numpy(float)[valid]
        return np.full(len(ppg), np.nan)

    gsr_cond = _load_col(cfg_data["gsr_columns"]["conductance"])
    gsr_res  = _load_col(cfg_data["gsr_columns"]["resistance"])

    n = len(ppg)
    duration_s = float(time_s[-1] - time_s[0])

    return SessionData(
        ppg=ppg,
        timestamps_ms=timestamps_ms,
        time_s=time_s,
        accel_ln=accel_ln,
        accel_wr=accel_wr,
        gsr_conductance=gsr_cond,
        gsr_resistance=gsr_res,
        fs_measured=fs_measured,
        fs_nominal=fs_nominal,
        ppg_column_used=ppg_col,
        source_path=str(csv_path),
        n_samples=n,
        duration_s=duration_s,
        metadata={
            "subject_folder": csv_path.parents[1].name,
            "session_name": csv_path.stem,
            "fs_source": fs_source,  # "measured" | "fallback_nominal" | "truncated"
        },
    )


# ── Batch discovery ────────────────────────────────────────────────────────────

def discover_csv_files(root_dir: Path, glob: str = "**/*-csv/*.csv") -> list[Path]:
    """
    Return sorted list of all session CSV paths under root_dir.

    Parameters
    ----------
    root_dir : Path
    glob : str
        Glob pattern relative to root_dir.

    Returns
    -------
    list[Path]  Sorted by (subject_folder, session_name).
    """
    files = sorted(root_dir.glob(glob))
    return files
