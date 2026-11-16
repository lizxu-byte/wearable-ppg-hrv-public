"""
peaks.py — PPG systolic peak detection methods

Peak detection is the most critical step in the PPG pipeline because every
downstream metric (IBI, HR, RMSSD, LF/HF) depends directly on the accuracy
of detected peak times.

Error propagation (Algorithm Engineering Discipline Rule 2):
    Peak timing error ε (ms)
    → IBI error ≈ 2ε (each IBI involves two consecutive peaks)
    → RMSSD error ≈ √2 × 2ε = 2√2 × ε
    Example: 5 ms peak timing MAE → ~14 ms RMSSD error
    This is why peak-timing MAE is the primary metric here, not HR MAE.

Three methods implemented:
    A. elgendi    — Elgendi 2013 dual-moving-average (via neurokit2)
    B. bishop     — Bishop & Ercole 2018 multi-scale (via neurokit2)
    C. prominence — scipy find_peaks with prominence-based adaptive threshold

OFFLINE note: all three methods are offline (process full signal at once).
For real-time use, elgendi can be made causal with a sliding window.

References:
    A. Elgendi M et al. (2013). Systolic Peak Detection in Acceleration
       Photoplethysmograms Measured from Emergency Responders in Tropical
       Conditions. PLoS ONE 8(10):e76585.
       https://doi.org/10.1371/journal.pone.0076585
       (Dual moving-average method; W1=systolic window, W2=beat window)

    B. Bishop SM, Ercole A (2018). Multi-scale peak and trough detection
       optimised for periodic and quasi-periodic neuroscience data.
       Front Neuroinform 12:62.
       https://doi.org/10.3389/fninf.2018.00062
       (Multi-scale approach robust to amplitude variation)

    C. Makowski D et al. (2021). NeuroKit2: A Python toolbox for neurophysio-
       logical signal processing. Behav Res Methods 53:1689-1696.
       https://doi.org/10.3758/s13428-020-01516-y
       (neurokit2 ppg_findpeaks wrapper)

    D. Allen J (2007). Photoplethysmography and its application in clinical
       physiological measurement. Physiol Meas 28(3):R1-R39.
       https://doi.org/10.1088/0967-3334/28/3/R01
       (±100 ms tolerance window standard for PPG peak matching)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.signal import find_peaks
import neurokit2 as nk


# ── Method A: Elgendi 2013 (via neurokit2) ────────────────────────────────────

def detect_elgendi(
    signal: np.ndarray,
    fs: int,
    peakwindow: float = 0.111,
    beatwindow: float = 0.667,
    beatoffset: float = 0.02,
    mindelay: float = 0.3,
) -> np.ndarray:
    """
    Elgendi 2013 dual-moving-average systolic peak detector.

    Squares the signal, applies two moving averages (MA_peak for systolic
    width, MA_beat for beat-level envelope), and detects peaks where the
    signal exceeds the beat-level threshold by beatoffset.

    Reference: Elgendi et al. (2013) PLoS ONE 8(10):e76585.

    Parameters
    ----------
    signal     : np.ndarray  Filtered PPG signal (mV), 1-D.
    fs         : int         Sampling rate (Hz).
    peakwindow : float       W1 — systolic peak duration (s). Default 0.111 s.
                             ≈ duration of systolic peak at normal HR.
                             For tachycardia (>100 bpm): consider reducing to 0.08 s.
    beatwindow : float       W2 — beat window (s). Default 0.667 s.
                             ≈ longest expected RR / 1.5. For bradycardia: increase.
    beatoffset : float       Offset above MA_beat to set threshold. Default 0.02.
    mindelay   : float       Minimum inter-peak interval (s). Default 0.3 s (=200 bpm).

    Returns
    -------
    np.ndarray  1-D array of peak sample indices (int).
    """
    signal = np.asarray(signal, dtype=float)
    peaks_dict = nk.ppg_findpeaks(
        signal, sampling_rate=fs, method="elgendi",
        show=False,
        peakwindow=peakwindow,
        beatwindow=beatwindow,
        beatoffset=beatoffset,
        mindelay=mindelay,
    )
    return np.asarray(peaks_dict.get("PPG_Peaks", []), dtype=int)


# ── Method B: Bishop & Ercole 2018 (via neurokit2) ───────────────────────────

def detect_bishop(
    signal: np.ndarray,
    fs: int,
) -> np.ndarray:
    """
    Bishop & Ercole 2018 multi-scale peak detector.

    Uses multiple temporal scales to find peaks that are consistently
    prominent across scales — robust to amplitude variation and noise.

    Reference: Bishop & Ercole (2018) Front Neuroinform 12:62.

    Parameters
    ----------
    signal : np.ndarray  Filtered PPG signal (mV), 1-D.
    fs     : int         Sampling rate (Hz).

    Returns
    -------
    np.ndarray  1-D array of peak sample indices (int).
    """
    signal = np.asarray(signal, dtype=float)
    peaks_dict = nk.ppg_findpeaks(signal, sampling_rate=fs, method="bishop", show=False)
    return np.asarray(peaks_dict.get("PPG_Peaks", []), dtype=int)


# ── Method C: Prominence-based adaptive threshold (scipy) ────────────────────

def detect_prominence(
    signal: np.ndarray,
    fs: int,
    min_distance_s: float = 0.3,
    prominence_factor: float = 0.3,
) -> np.ndarray:
    """
    Prominence-based adaptive peak detector using scipy.signal.find_peaks.

    Sets minimum prominence as a fraction of the signal's peak-to-peak range,
    making the threshold adaptive to signal amplitude changes. More robust than
    a fixed absolute threshold when signal amplitude drifts.

    Parameters
    ----------
    signal             : np.ndarray  Filtered PPG signal (mV), 1-D.
    fs                 : int         Sampling rate (Hz).
    min_distance_s     : float       Minimum inter-peak interval (s). Default 0.3.
                                     Corresponds to 200 bpm ceiling.
    prominence_factor  : float       Min prominence = factor × signal p-p range.
                                     Default 0.3 (30% of range).
                                     For low-amplitude finger PPG at rest: 0.2–0.3.
                                     For noisy signals: increase to 0.4.

    Returns
    -------
    np.ndarray  1-D array of peak sample indices (int).
    """
    signal = np.asarray(signal, dtype=float)
    min_distance_samples = int(min_distance_s * fs)
    pp_range = float(np.percentile(signal, 98) - np.percentile(signal, 2))
    min_prominence = prominence_factor * pp_range

    peaks, _ = find_peaks(
        signal,
        distance=min_distance_samples,
        prominence=min_prominence,
    )
    return peaks.astype(int)


# ── Dicrotic wave deduplication ───────────────────────────────────────────────

def _dedup_peaks(
    peaks: np.ndarray,
    signal: np.ndarray,
    fs: int,
    max_check_s: float = 0.60,
) -> np.ndarray:
    """
    Remove dicrotic wave false detections by comparing consecutive peak pairs.

    The dicrotic wave (secondary notch) in fingertip PPG occurs at 40–70% of
    the RR interval and has ~20–30% of systolic peak amplitude. Detection
    methods that use only local prominence (Bishop, scipy prominence) mistake
    it for a systolic peak under noise or at higher heart rates.

    Algorithm: for every consecutive pair of detected peaks within max_check_s,
    keep only the taller peak (systolic > dicrotic by design).

    Limitation: valid at HR ≤ ~100 bpm. At HR > 100 bpm, two consecutive
    systolic peaks may be < 600 ms apart — dedup would incorrectly compare them.
    For tachycardia populations, reduce max_check_s to 0.45 s (133 bpm ceiling).

    Reference: Allen J (2007) Physiol Meas 28(3):R1-R39.
               (Dicrotic notch morphology in fingertip PPG, Section 2.3)

    Parameters
    ----------
    peaks       : np.ndarray  Detected peak indices (may include dicrotic waves).
    signal      : np.ndarray  The PPG signal (used for amplitude comparison).
    fs          : int         Sampling rate (Hz).
    max_check_s : float       Pairs within this window are compared. Default 0.6 s.
                              Covers dicrotic waves for HR ≤ 100 bpm.
                              For wrist/earlobe PPG with weaker dicrotic: 0.45 s.

    Returns
    -------
    np.ndarray  Deduplicated peak indices.
    """
    if len(peaks) < 2:
        return peaks

    peaks = np.sort(peaks)
    max_check_samples = int(max_check_s * fs)
    keep = np.ones(len(peaks), dtype=bool)

    i = 0
    while i < len(peaks):
        if not keep[i]:
            i += 1
            continue
        j = i + 1
        while j < len(peaks) and (peaks[j] - peaks[i]) <= max_check_samples:
            if keep[j]:
                if signal[peaks[i]] >= signal[peaks[j]]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
            j += 1
        i += 1

    return peaks[keep]


# ── Dispatcher ────────────────────────────────────────────────────────────────

_METHODS = {
    "elgendi":    detect_elgendi,
    "bishop":     detect_bishop,
    "prominence": detect_prominence,
}


def detect_peaks(
    signal: np.ndarray,
    fs: int,
    method: str,
    dedup: bool = True,
    dedup_max_check_s: float = 0.60,
    **kwargs,
) -> np.ndarray:
    """
    Dispatch to the selected peak detection method, with optional dicrotic
    wave deduplication.

    Parameters
    ----------
    signal           : np.ndarray  Filtered PPG signal (mV).
    fs               : int         Sampling rate (Hz).
    method           : str         One of "elgendi", "bishop", "prominence".
    dedup            : bool        Apply dicrotic wave deduplication. Default True.
    dedup_max_check_s: float       Dedup window (s). Default 0.6 s (HR ≤ 100 bpm).
    **kwargs                       Forwarded to the method function.

    Returns
    -------
    np.ndarray  Peak sample indices.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown peak method '{method}'. Choose from {list(_METHODS)}")
    peaks = _METHODS[method](signal, fs, **kwargs)
    if dedup:
        peaks = _dedup_peaks(peaks, signal, fs, max_check_s=dedup_max_check_s)
    return peaks


# ── Evaluation metrics ────────────────────────────────────────────────────────

def evaluate(
    detected: np.ndarray,
    true_peaks: np.ndarray,
    fs: int,
    tolerance_ms: float = 100.0,
) -> dict:
    """
    Compute detection quality metrics against ground-truth peaks.

    Matching rule: a detected peak is a True Positive (TP) if it falls within
    ±tolerance_ms of a true peak. Each true peak can be matched at most once
    (greedy nearest-neighbor matching).

    Standard: Allen J (2007) Physiol Meas 28(3):R1-R39.
    Tolerance ±100 ms is the accepted standard for PPG peak matching.

    Parameters
    ----------
    detected    : np.ndarray  Detected peak indices.
    true_peaks  : np.ndarray  Ground-truth peak indices.
    fs          : int         Sampling rate (Hz).
    tolerance_ms: float       Match tolerance (ms). Default 100 ms.

    Returns
    -------
    dict with keys:
        tp, fp, fn          — counts
        sensitivity         — TP / (TP + FN)
        ppv                 — TP / (TP + FP)  [positive predictive value]
        f1                  — harmonic mean of sensitivity and PPV
        peak_timing_mae_ms  — mean |detected - true| for matched pairs (ms)
        peak_timing_std_ms  — std of timing errors for matched pairs (ms)
        rmssd_true_ms       — RMSSD of ground-truth RR intervals
        rmssd_detected_ms   — RMSSD of detected RR intervals (all detected peaks)
        rmssd_error_ms      — |rmssd_detected - rmssd_true|
    """
    tol_samples = tolerance_ms / 1000.0 * fs
    detected  = np.sort(np.asarray(detected,   dtype=int))
    true_peaks = np.sort(np.asarray(true_peaks, dtype=int))

    matched_true = set()
    matched_det  = set()
    timing_errors = []

    for d_idx in detected:
        diffs = np.abs(true_peaks.astype(float) - float(d_idx))
        best = int(np.argmin(diffs))
        if diffs[best] <= tol_samples and best not in matched_true:
            matched_true.add(best)
            matched_det.add(int(np.where(detected == d_idx)[0][0]))
            timing_errors.append(diffs[best] / fs * 1000.0)

    tp = len(matched_true)
    fp = len(detected) - tp
    fn = len(true_peaks) - tp

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = (2 * sensitivity * ppv / (sensitivity + ppv)
                   if (sensitivity + ppv) > 0 else 0.0)

    peak_timing_mae_ms = float(np.mean(timing_errors))  if timing_errors else float("nan")
    peak_timing_std_ms = float(np.std(timing_errors))   if len(timing_errors) > 1 else float("nan")

    # RMSSD
    def rmssd(peaks):
        if len(peaks) < 2:
            return float("nan")
        rr = np.diff(peaks.astype(float)) / fs * 1000.0
        return float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else float("nan")

    rmssd_true     = rmssd(true_peaks)
    rmssd_detected = rmssd(detected)
    rmssd_error    = abs(rmssd_detected - rmssd_true) if np.isfinite(rmssd_detected) else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "sensitivity":         sensitivity,
        "ppv":                 ppv,
        "f1":                  f1,
        "peak_timing_mae_ms":  peak_timing_mae_ms,
        "peak_timing_std_ms":  peak_timing_std_ms,
        "rmssd_true_ms":       rmssd_true,
        "rmssd_detected_ms":   rmssd_detected,
        "rmssd_error_ms":      rmssd_error,
    }


# ── Comparison runner ─────────────────────────────────────────────────────────

def compare_methods(
    signal: np.ndarray,
    true_peaks: np.ndarray,
    fs: int,
    configs: list[dict],
    tolerance_ms: float = 100.0,
) -> list[dict]:
    """
    Run multiple peak detection methods and return evaluation metrics for each.

    Parameters
    ----------
    signal       : np.ndarray  Filtered PPG signal.
    true_peaks   : np.ndarray  Ground-truth peak indices.
    fs           : int         Sampling rate (Hz).
    configs      : list[dict]  Each dict: {name, method, **kwargs}.
    tolerance_ms : float       Matching tolerance (ms).

    Returns
    -------
    list[dict]  One result dict per method with evaluation metrics.
    """
    results = []
    for cfg in configs:
        name   = cfg["name"]
        method = cfg["method"]
        kwargs = {k: v for k, v in cfg.items() if k not in ("name", "method")}
        row = {"name": name, "method": method}
        try:
            detected = detect_peaks(signal, fs, method, **kwargs)
            row.update(evaluate(detected, true_peaks, fs, tolerance_ms))
            row["n_detected"] = len(detected)
        except Exception as e:
            row["error"] = str(e)
        results.append(row)
    return results
