"""
hrv_time.py — HRV time-domain metrics (Task Force 1996 compliant)

All metrics follow the definitions in:
    Task Force of the European Society of Cardiology and the North American
    Society of Pacing and Electrophysiology (1996). Heart rate variability:
    standards of measurement, physiological interpretation and clinical use.
    Circulation 93:1043-1065.
    https://doi.org/10.1161/01.CIR.93.5.1043

Known bug in old code (V1) — FIXED HERE:
    pNN50 denominator must be len(diff_rr), NOT len(rr_ms).
    Task Force 1996 (p.1048): "the proportion derived by dividing NN50 by
    the total number of NN intervals."
    "Total number of NN intervals" = len(diff_rr) = N-1, not N.
    Old code used len(rr_ms) = N, producing pNN50 that is systematically
    low by factor (N-1)/N ≈ 0.3% for a 5-min recording (~350 beats).

Metric definitions (Task Force 1996):
    MeanNN  : mean of all NN intervals (ms)
    SDNN    : standard deviation of all NN intervals (ms); sample std, ddof=1
    RMSSD   : root mean square of successive RR differences (ms)
    NN50    : count of successive RR differences > 50 ms
    pNN50   : NN50 / (N-1) × 100  [%] — N = number of NN intervals
    MeanHR  : mean heart rate (bpm) = 60000 / MeanNN
    MinHR   : minimum HR (bpm) = 60000 / max(NN)   [slowest beat → lowest HR]
    MaxHR   : maximum HR (bpm) = 60000 / min(NN)   [fastest beat → highest HR]
    CVNN    : coefficient of variation = SDNN / MeanNN × 100  [%]
    CVSD    : coefficient of variation of successive diffs = RMSSD / MeanNN × 100
"""

from __future__ import annotations

import numpy as np


def compute_hrv_time(rr_ms: np.ndarray) -> dict:
    """
    Compute all standard HRV time-domain metrics from a clean RR interval array.

    Reference: Task Force ESC/NASPE (1996) Circulation 93:1043-1065.

    Parameters
    ----------
    rr_ms : np.ndarray  Clean RR intervals (ms), 1-D. Must contain ≥ 2 values.
                        All values should be within physiological range
                        (300–2000 ms); outliers should be removed before
                        calling this function (rr_clean.py).

    Returns
    -------
    dict with keys:
        n_beats     — number of NN intervals used
        mean_nn_ms  — MeanNN (ms)
        sdnn_ms     — SDNN (ms), sample std (ddof=1)
        rmssd_ms    — RMSSD (ms)
        nn50        — count of |ΔRR| > 50 ms
        pnn50_pct   — pNN50 (%), denominator = N-1 per Task Force 1996
        mean_hr_bpm — MeanHR (bpm)
        min_hr_bpm  — MinHR (bpm) = 60000 / max(RR)
        max_hr_bpm  — MaxHR (bpm) = 60000 / min(RR)
        cvnn_pct    — CVNN (%) = SDNN / MeanNN × 100
        cvsd_pct    — CVSD (%) = RMSSD / MeanNN × 100

    Raises
    ------
    ValueError  if rr_ms has fewer than 2 elements.
    """
    rr_ms = np.asarray(rr_ms, dtype=float)
    if len(rr_ms) < 2:
        raise ValueError(f"Need ≥ 2 RR intervals; got {len(rr_ms)}")
    if np.any(rr_ms <= 0):
        raise ValueError("RR intervals must be positive (non-zero)")

    diff_rr = np.diff(rr_ms)       # successive differences, length = N-1
    n        = len(rr_ms)          # N (number of NN intervals)
    n_diff   = len(diff_rr)        # N-1

    mean_nn = float(np.mean(rr_ms))
    sdnn    = float(np.std(rr_ms, ddof=1))
    rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))

    nn50    = int(np.sum(np.abs(diff_rr) > 50.0))
    # Task Force 1996: denominator = N-1 (total number of NN intervals
    # that CAN form a successive difference pair)
    pnn50   = (nn50 / n_diff * 100.0) if n_diff > 0 else float("nan")

    mean_hr = 60000.0 / mean_nn
    min_hr  = 60000.0 / float(np.max(rr_ms))
    max_hr  = 60000.0 / float(np.min(rr_ms))

    cvnn = (sdnn   / mean_nn * 100.0) if mean_nn > 0 else float("nan")
    cvsd = (rmssd  / mean_nn * 100.0) if mean_nn > 0 else float("nan")

    return {
        "n_beats":     n,
        "mean_nn_ms":  mean_nn,
        "sdnn_ms":     sdnn,
        "rmssd_ms":    rmssd,
        "nn50":        nn50,
        "pnn50_pct":   pnn50,
        "mean_hr_bpm": mean_hr,
        "min_hr_bpm":  min_hr,
        "max_hr_bpm":  max_hr,
        "cvnn_pct":    cvnn,
        "cvsd_pct":    cvsd,
    }
