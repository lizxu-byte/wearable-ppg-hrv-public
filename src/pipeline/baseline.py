"""
baseline.py — Baseline drift removal methods for PPG signals

Three methods are implemented and can be compared head-to-head:

  A. polynomial_detrend  — fit and subtract a low-order polynomial trend
  B. highpass_filter     — zero-phase Butterworth high-pass (most common)
  C. moving_median       — estimate baseline as moving-median, then subtract

All methods accept the same interface: (signal, fs, **kwargs) → detrended_signal.
Parameter values come from config.yaml; no magic numbers in this file.

Key downstream risk (Algorithm Engineering Discipline Rule 4):
    Residual baseline causes the peak detector to fire on drift slopes instead
    of systolic peaks → wrong IBI → RMSSD corrupted even if HR MAE looks fine.
    Validation therefore reports BOTH residual-baseline RMSE and peak-timing
    error, not only HR MAE.

References:
    A. Tarvainen MP et al. (2002). An advanced detrending method with
       application to HRV analysis. IEEE Trans Biomed Eng 49(2):172-175.
       https://doi.org/10.1109/10.979357
       (Polynomial / smoothness-prior detrending for biosignals)

    B. Elgendi M et al. (2013). Systolic Peak Detection in Acceleration
       Photoplethysmograms. PLoS ONE 8(10):e76585.
       https://doi.org/10.1371/journal.pone.0076585
       (Recommends 0.5 Hz high-pass before peak detection)

    C. Orphanidou C et al. (2015). Signal-Quality Indices for the ECG and PPG.
       IEEE J Biomed Health Inform 19(3):832-838.
       https://doi.org/10.1109/JBHI.2014.2338351
       (Moving-median baseline used as preprocessing step in SQI pipeline)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, medfilt


# ── Method A: Polynomial detrend ──────────────────────────────────────────────

def polynomial_detrend(
    signal: np.ndarray,
    fs: int,
    order: int = 3,
) -> np.ndarray:
    """
    Remove low-frequency trend by subtracting a fitted polynomial.

    Fits a polynomial of given order to the signal and subtracts it.
    Simple and fast; works well when drift is smooth and monotone.
    May underperform when drift has abrupt changes (motion artefacts).

    Reference: Tarvainen et al. (2002) IEEE Trans Biomed Eng 49(2):172-175.

    Parameters
    ----------
    signal : np.ndarray
        Raw PPG signal (mV), 1-D.
    fs : int
        Sampling rate (Hz). Not used directly but kept for API consistency.
    order : int
        Polynomial order. Default 3.
        Higher order → can track faster drift, but risks removing signal.
        If population has faster respiration (children): consider order 4–5.

    Returns
    -------
    np.ndarray
        Detrended signal (mV), same length as input.
    """
    signal = np.asarray(signal, dtype=float)
    t = np.arange(len(signal))
    coeffs = np.polyfit(t, signal, order)
    trend = np.polyval(coeffs, t)
    return signal - trend


# ── Method B: High-pass Butterworth (zero-phase) ─────────────────────────────

def highpass_filter(
    signal: np.ndarray,
    fs: int,
    cutoff_hz: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """
    Remove baseline drift with a zero-phase Butterworth high-pass filter.

    Uses sosfiltfilt (second-order sections + forward-backward pass) for
    numerical stability and zero phase delay. This is an OFFLINE / non-causal
    method — it requires the full signal and cannot be used for real-time
    streaming without modification.

    For real-time use: replace sosfiltfilt with sosfilt and accept group-delay
    latency of ~(order / (2 * cutoff_hz)) seconds.

    Reference: Elgendi et al. (2013) PLoS ONE 8(10):e76585.
               Recommends ≥0.5 Hz high-pass before systolic peak detection.

    Parameters
    ----------
    signal : np.ndarray
        Raw PPG signal (mV), 1-D.
    fs : int
        Sampling rate (Hz).
    cutoff_hz : float
        High-pass cutoff frequency (Hz). Default 0.5.
        Lower bound set by cardiac fundamental (~0.4 Hz = 24 bpm).
        If studying bradycardia populations: consider 0.3–0.4 Hz.
    order : int
        Filter order. Default 4.
        Higher order → steeper roll-off but more ringing near transients.

    Returns
    -------
    np.ndarray
        High-pass filtered signal (mV), offline (zero-phase).
    """
    signal = np.asarray(signal, dtype=float)
    nyq = fs / 2.0
    norm_cutoff = cutoff_hz / nyq
    sos = butter(order, norm_cutoff, btype="high", output="sos")
    return sosfiltfilt(sos, signal)


# ── Method C: Moving-median baseline subtraction ─────────────────────────────

def moving_median(
    signal: np.ndarray,
    fs: int,
    window_s: float = 2.0,
) -> np.ndarray:
    """
    Estimate and subtract a moving-median baseline.

    The moving median with a window longer than one cardiac cycle serves as
    a robust baseline estimator: it is insensitive to sharp PPG peaks (unlike
    moving mean) while tracking slow drift.

    Window must be longer than the longest expected RR interval (1500 ms = 1.5 s).
    Default 2.0 s corresponds to a minimum HR of 30 bpm (2000 ms period).
    For very low HR populations (athletes at rest): increase to 3.0 s.

    Reference: Orphanidou et al. (2015) IEEE J Biomed Health Inform 19(3):832-838.

    Parameters
    ----------
    signal : np.ndarray
        Raw PPG signal (mV), 1-D.
    fs : int
        Sampling rate (Hz).
    window_s : float
        Median filter window length (seconds). Default 2.0.
        Rule: window_s > max_expected_rr_ms / 1000. For 40 bpm: 1.5 s.
        Current sensor position: fingertip. Finger PPG has stronger respiratory
        baseline drift than wrist; 2.0 s is appropriate.
        If sensor moves to wrist or elderly population: increase to 2.5–3.0 s.

    Returns
    -------
    np.ndarray
        Detrended signal (mV): original minus moving-median baseline.
    """
    signal = np.asarray(signal, dtype=float)
    kernel = int(round(window_s * fs))
    if kernel % 2 == 0:
        kernel += 1  # medfilt requires odd kernel size
    baseline = medfilt(signal, kernel_size=kernel)
    return signal - baseline


# ── Dispatcher ────────────────────────────────────────────────────────────────

_METHODS = {
    "polynomial_detrend": polynomial_detrend,
    "highpass_filter":    highpass_filter,
    "moving_median":      moving_median,
}


def remove_baseline(
    signal: np.ndarray,
    fs: int,
    method: str,
    **kwargs,
) -> np.ndarray:
    """
    Dispatch to the selected baseline removal method.

    Parameters
    ----------
    signal : np.ndarray  Raw PPG (mV).
    fs     : int         Sampling rate (Hz).
    method : str         One of "polynomial_detrend", "highpass_filter",
                         "moving_median".
    **kwargs             Forwarded to the method function.

    Returns
    -------
    np.ndarray  Detrended signal (mV).
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown baseline method '{method}'. "
                         f"Choose from {list(_METHODS)}")
    return _METHODS[method](signal, fs, **kwargs)


# ── Comparison helper ─────────────────────────────────────────────────────────

def compare_methods(
    signal_noisy: np.ndarray,
    signal_clean: np.ndarray,
    true_peak_indices: np.ndarray,
    fs: int,
    configs: dict,
) -> dict:
    """
    Run all baseline methods on signal_noisy and report quantitative metrics.

    Metrics reported per method (Algorithm Engineering Discipline Rule 4:
    must report metrics that CAN expose failure, not only HR MAE):

      residual_rmse     — RMSE between detrended signal and clean signal
                          (measures remaining baseline + waveform distortion)
      peak_timing_mae_ms— Mean absolute error of detected peak times vs true
                          peaks (uses argmax in ±50 ms window around true peak)
      waveform_rmse     — RMSE computed only at samples within ±2σ of each
                          true peak (measures peak-shape distortion only)

    Parameters
    ----------
    signal_noisy      : np.ndarray  Signal with baseline drift (no AWGN ideal).
    signal_clean      : np.ndarray  Ground-truth clean PPG.
    true_peak_indices : np.ndarray  Ground-truth systolic peak sample indices.
    fs                : int         Sampling rate (Hz).
    configs           : dict        {method_name: {param: value, ...}}
                        Example: {"highpass_filter": {"cutoff_hz": 0.5, "order": 4}}

    Returns
    -------
    dict  {method_name: {metric_name: value}}
    """
    results = {}

    for method, params in configs.items():
        try:
            detrended = remove_baseline(signal_noisy, fs, method, **params)
        except Exception as e:
            results[method] = {"error": str(e)}
            continue

        # ── Residual RMSE (whole signal) ──────────────────────────────────
        residual_rmse = float(np.sqrt(np.mean((detrended - signal_clean) ** 2)))

        # ── Peak-timing MAE ───────────────────────────────────────────────
        # For each true peak, find argmax in ±50 ms window in detrended signal
        half_win = int(0.05 * fs)  # 50 ms
        n = len(detrended)
        timing_errors_ms = []
        for idx in true_peak_indices:
            lo = max(0, idx - half_win)
            hi = min(n, idx + half_win + 1)
            detected_offset = np.argmax(detrended[lo:hi])
            detected_idx = lo + detected_offset
            timing_errors_ms.append(abs(detected_idx - idx) / fs * 1000)

        peak_timing_mae_ms = float(np.mean(timing_errors_ms)) if timing_errors_ms else float("nan")

        # ── Waveform RMSE at peak regions ─────────────────────────────────
        # Window: ±2σ around each peak, where σ ≈ 0.04 * mean_RR
        if len(true_peak_indices) > 1:
            mean_rr_samples = float(np.mean(np.diff(true_peak_indices)))
            sigma = int(round(0.04 * mean_rr_samples))
        else:
            sigma = int(0.034 * fs)  # fallback ~34 ms at 256 Hz

        peak_samples = []
        for idx in true_peak_indices:
            lo = max(0, idx - 2 * sigma)
            hi = min(n, idx + 2 * sigma + 1)
            peak_samples.extend(range(lo, hi))
        peak_samples = np.unique(peak_samples)

        if len(peak_samples) > 0:
            waveform_rmse = float(np.sqrt(np.mean(
                (detrended[peak_samples] - signal_clean[peak_samples]) ** 2
            )))
        else:
            waveform_rmse = float("nan")

        results[method] = {
            "residual_rmse":      residual_rmse,
            "peak_timing_mae_ms": peak_timing_mae_ms,
            "waveform_rmse":      waveform_rmse,
        }

    return results
