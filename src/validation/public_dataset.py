"""
public_dataset.py — BIDMC PPG and Respiration Dataset reader

Provides a clean interface to the BIDMC dataset (PhysioNet) for
end-to-end algorithm validation with ground-truth ECG R-peaks.

Dataset:
    Pimentel MAF et al. (2017). Toward a robust estimation of respiratory
    rate from pulse oximeters. IEEE Trans Biomed Eng 64(8):1914-1923.
    https://doi.org/10.1109/TBME.2016.2613124

    Available at: https://physionet.org/content/bidmc/
    53 subjects, 8 min each, 125 Hz.
    Signals: PLETH (PPG, NU), II (ECG lead II, mV), RESP (respiration, pm).

Ground truth R-peaks:
    Detected from ECG lead II using neurokit2 ecg_peaks() (Hamilton 2002
    detector, improved by Engzee-Corrected method in neurokit2 defaults).
    This is appropriate because:
    - BIDMC ECG is clean (ICU setting, high SNR)
    - neurokit2 ECG detector validated on MIT-BIH (F1 > 0.99)
    - True R-peak timing error < 4 ms at 125 Hz (1 sample = 8 ms)

Error propagation (Algorithm Engineering Discipline Rule 2):
    ECG R-peak timing → true RR intervals → true HR and RMSSD reference.
    Error in ECG detection propagates to all PPG comparison metrics.
    Mitigation: use neurokit2 default detector + quality filter (remove
    beats with HR outside [30, 200] bpm or RR change > 50%).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class BIDMCRecord:
    """Single BIDMC recording with signals and ground-truth annotations."""
    record_id:    str           # e.g. "bidmc01"
    fs:           int           # 125 Hz
    n_samples:    int
    duration_s:   float
    ppg:          np.ndarray   # PLETH signal (NU), shape (n_samples,)
    ecg:          np.ndarray   # ECG lead II (mV), shape (n_samples,)
    resp:         np.ndarray   # Respiratory signal (pm), shape (n_samples,)
    r_peaks:      np.ndarray   # ECG R-peak indices (samples), 1-D int array
    rr_ms:        np.ndarray   # RR intervals from ECG peaks (ms)
    hr_bpm:       float        # Mean HR from ECG (bpm)
    rmssd_ms:     float        # RMSSD from ECG RR series (ms)


# ── Record reader ─────────────────────────────────────────────────────────────

def load_bidmc_record(
    record_path: str | Path,
    detect_peaks: bool = True,
) -> BIDMCRecord:
    """
    Load a single BIDMC WFDB record and optionally detect ECG R-peaks.

    Parameters
    ----------
    record_path : str | Path
        Path to the record WITHOUT extension (e.g. /data/bidmc/bidmc01).
    detect_peaks : bool
        If True, detect ECG R-peaks via neurokit2 and compute RR / HR / RMSSD.
        Set False for fast loading when only the raw signals are needed.

    Returns
    -------
    BIDMCRecord  dataclass with ppg, ecg, r_peaks, rr_ms, hr_bpm, rmssd_ms.

    Raises
    ------
    FileNotFoundError  if the .hea file does not exist.
    ValueError         if required signal columns are missing from the record.
    """
    record_path = Path(record_path)
    hea = record_path.with_suffix(".hea")
    if not hea.exists():
        raise FileNotFoundError(f"BIDMC header not found: {hea}")

    rec = wfdb.rdrecord(str(record_path))

    # Locate signal columns by name (trailing commas in BIDMC names)
    sig_names = [s.rstrip(",").strip().upper() for s in rec.sig_name]

    def _get_col(name: str) -> np.ndarray:
        try:
            idx = sig_names.index(name)
        except ValueError:
            raise ValueError(
                f"Signal '{name}' not found in {record_path.name}. "
                f"Available: {sig_names}"
            )
        col = rec.p_signal[:, idx].astype(float)
        return col

    ppg  = _get_col("PLETH")
    ecg  = _get_col("II")
    resp = _get_col("RESP")

    r_peaks  = np.array([], dtype=int)
    rr_ms    = np.array([], dtype=float)
    hr_bpm   = float("nan")
    rmssd_ms = float("nan")

    if detect_peaks:
        r_peaks, rr_ms, hr_bpm, rmssd_ms = _ecg_peaks_and_hrv(ecg, rec.fs)

    return BIDMCRecord(
        record_id  = record_path.stem,
        fs         = int(rec.fs),
        n_samples  = rec.sig_len,
        duration_s = rec.sig_len / rec.fs,
        ppg        = ppg,
        ecg        = ecg,
        resp       = resp,
        r_peaks    = r_peaks,
        rr_ms      = rr_ms,
        hr_bpm     = hr_bpm,
        rmssd_ms   = rmssd_ms,
    )


# ── ECG peak detection ────────────────────────────────────────────────────────

def _ecg_peaks_and_hrv(
    ecg: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Detect R-peaks from ECG and compute mean HR and RMSSD.

    Uses neurokit2 Hamilton (2002) detector, validated on MIT-BIH (F1 > 0.99).

    Reference:
        Hamilton PS (2002). Open source ECG analysis software documentation.
        EP Limited. (Implemented in neurokit2 as method='hamilton2002')

    Parameters
    ----------
    ecg : np.ndarray  ECG signal (mV or arbitrary units), 1-D.
    fs  : int         Sampling rate (Hz).

    Returns
    -------
    r_peaks  : np.ndarray  int, sample indices of R-peaks
    rr_ms    : np.ndarray  float, RR intervals (ms) between consecutive peaks
    hr_bpm   : float  mean HR (bpm) from ECG
    rmssd_ms : float  RMSSD of ECG RR series (ms)
    """
    import neurokit2 as nk

    _, info = nk.ecg_peaks(ecg, sampling_rate=fs, method="hamilton2002",
                           correct_artifacts=True)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)

    if len(peaks) < 2:
        return peaks, np.array([]), float("nan"), float("nan")

    rr_ms = np.diff(peaks) / fs * 1000.0

    # Physiological RR filter: remove impossible beats (HR < 30 or > 200 bpm)
    # and sudden jumps > 50% of the preceding interval (artifact / ectopic)
    valid = (rr_ms > 300.0) & (rr_ms < 2000.0)
    if len(rr_ms) > 1:
        rr_ratio = np.abs(np.diff(rr_ms)) / rr_ms[:-1]
        valid[1:] &= rr_ratio < 0.50
    rr_clean = rr_ms[valid]

    hr_bpm   = 60_000.0 / float(np.mean(rr_clean)) if len(rr_clean) > 0 else float("nan")
    rmssd_ms = float(np.sqrt(np.mean(np.diff(rr_clean) ** 2))) if len(rr_clean) > 1 else float("nan")

    return peaks, rr_ms, hr_bpm, rmssd_ms


# ── Batch loader ──────────────────────────────────────────────────────────────

def load_all_bidmc(
    data_dir: str | Path,
    n_records: int | None = None,
    detect_peaks: bool = True,
) -> list[BIDMCRecord]:
    """
    Load all (or first n_records) BIDMC records from data_dir.

    Parameters
    ----------
    data_dir    : str | Path  Directory containing bidmc01.hea, etc.
    n_records   : int | None  If given, load only the first n_records alphabetically.
    detect_peaks: bool        Whether to run ECG peak detection for each record.

    Returns
    -------
    list[BIDMCRecord]  Sorted by record_id.

    Notes
    -----
    Records that fail to parse are skipped with a warning printed to stderr.
    """
    data_dir = Path(data_dir)
    hea_files = sorted(data_dir.glob("bidmc*.hea"))
    if n_records is not None:
        hea_files = hea_files[:n_records]

    records = []
    for hea in hea_files:
        record_path = hea.with_suffix("")
        try:
            rec = load_bidmc_record(record_path, detect_peaks=detect_peaks)
            records.append(rec)
        except Exception as exc:
            import sys
            print(f"[WARN] {hea.stem}: {exc}", file=sys.stderr)

    return records
