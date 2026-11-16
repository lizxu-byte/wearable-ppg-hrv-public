"""
filter.py — Bandpass filtering for PPG signals (Plan B: one-step)

A single bandpass filter simultaneously removes baseline drift (<lowcut Hz)
and high-frequency noise (>highcut Hz), preserving only the cardiac signal band.

This is the standard approach in PPG literature and replaces the two-step
"baseline removal then low-pass" pipeline for most use cases.

OFFLINE vs REAL-TIME:
    All functions here use sosfiltfilt (zero-phase, forward-backward pass).
    This is NON-CAUSAL — it requires the complete signal before processing.
    Latency = full signal length. Suitable for offline batch analysis.

    For real-time streaming: replace sosfiltfilt → sosfilt.
    Group delay ≈ order / (2 × lowcut_hz) seconds.
    Quantified difference on validation set: see test_filter.py results.

Sensor context:
    Fingertip PPG (Shimmer3 GSR+), 256 Hz nominal.
    Respiratory baseline drift is the dominant low-frequency artefact (0.1–0.4 Hz).
    Motion artefacts appear as broadband bursts handled separately in motion.py.

References:
    Elgendi M et al. (2013). Systolic Peak Detection in Acceleration
    Photoplethysmograms Measured from Emergency Responders in Tropical
    Conditions. PLoS ONE 8(10):e76585.
    https://doi.org/10.1371/journal.pone.0076585
    (Recommends 0.5–8 Hz bandpass for fingertip PPG peak detection)

    Charlton PH et al. (2022). Wearable Photoplethysmography for
    Cardiovascular Monitoring. Proc IEEE 110(3):355-381.
    https://doi.org/10.1109/JPROC.2022.3149785
    (Survey of PPG filtering practices; 0.5–8 Hz most common for HR/HRV)

    Proakis JG, Manolakis DG (2006). Digital Signal Processing, 4th ed.
    Prentice Hall. (Butterworth filter design theory, Chapter 10)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt, sosfreqz


# ── Core filter function ───────────────────────────────────────────────────────

def bandpass_butterworth(
    signal: np.ndarray,
    fs: int,
    lowcut_hz: float = 0.5,
    highcut_hz: float = 8.0,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Bandpass Butterworth filter for PPG signals.

    One-step replacement for separate baseline removal + low-pass filtering.
    Preserves cardiac signal band while removing respiratory baseline drift
    and high-frequency noise simultaneously.

    OFFLINE mode (zero_phase=True):  sosfiltfilt — zero phase delay, non-causal.
    REAL-TIME mode (zero_phase=False): sosfilt    — causal, group delay exists.

    Reference: Elgendi et al. (2013) PLoS ONE 8(10):e76585.
               Charlton et al. (2022) Proc IEEE 110(3):355-381.

    Parameters
    ----------
    signal     : np.ndarray  Raw PPG signal (mV), 1-D.
    fs         : int         Sampling rate (Hz). Shimmer3 nominal: 256.
    lowcut_hz  : float       High-pass edge (Hz). Default 0.5.
                             Lower bound set by cardiac fundamental (~40 bpm = 0.67 Hz;
                             0.5 Hz gives margin). For bradycardia studies: try 0.4 Hz.
                             Current sensor: fingertip — 0.5 Hz removes respiratory drift.
    highcut_hz : float       Low-pass edge (Hz). Default 8.0.
                             Upper bound: 3rd harmonic of max HR (~200 bpm → 10 Hz).
                             Elgendi (2013) recommends 8 Hz for systolic peak detection.
                             Stricter 5 Hz is used in some HRV-only pipelines.
    order      : int         Filter order (per high-pass and low-pass section).
                             Total filter order = 2 × order (bandpass = HP + LP).
                             Default 4. Higher → steeper roll-off but more ringing.
                             Ringing at peaks → peak-timing error → RMSSD distortion.
    zero_phase : bool        True = sosfiltfilt (offline, zero phase).
                             False = sosfilt (real-time, causal, has group delay).

    Returns
    -------
    np.ndarray  Filtered PPG signal (mV), same length as input.
    """
    signal = np.asarray(signal, dtype=float)
    nyq = fs / 2.0

    if lowcut_hz <= 0 or highcut_hz >= nyq:
        raise ValueError(
            f"Cutoff frequencies must satisfy 0 < lowcut ({lowcut_hz}) < "
            f"highcut ({highcut_hz}) < nyquist ({nyq})"
        )
    if lowcut_hz >= highcut_hz:
        raise ValueError(f"lowcut ({lowcut_hz}) must be < highcut ({highcut_hz})")

    sos = butter(
        order,
        [lowcut_hz / nyq, highcut_hz / nyq],
        btype="band",
        output="sos",
    )

    if zero_phase:
        return sosfiltfilt(sos, signal)
    else:
        return sosfilt(sos, signal)


# ── Frequency response analyser ───────────────────────────────────────────────

def frequency_response(
    fs: int,
    lowcut_hz: float,
    highcut_hz: float,
    order: int,
    n_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the magnitude frequency response of a bandpass Butterworth filter.

    Parameters
    ----------
    fs, lowcut_hz, highcut_hz, order : same as bandpass_butterworth.
    n_points : int  Number of frequency points.

    Returns
    -------
    freqs_hz : np.ndarray  Frequency axis (Hz).
    mag_db   : np.ndarray  Magnitude response (dB). 0 dB = no attenuation.
    """
    nyq = fs / 2.0
    sos = butter(
        order,
        [lowcut_hz / nyq, highcut_hz / nyq],
        btype="band",
        output="sos",
    )
    w, h = sosfreqz(sos, worN=n_points, fs=fs)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    return w, mag_db


def passband_ripple_db(
    freqs_hz: np.ndarray,
    mag_db: np.ndarray,
    lowcut_hz: float,
    highcut_hz: float,
) -> float:
    """
    Return peak-to-peak ripple (dB) within the passband [lowcut, highcut].

    A well-designed Butterworth filter has near-zero ripple in the passband.
    Quality gate: passband_ripple < 1 dB (Elgendi 2013 criterion).

    Parameters
    ----------
    freqs_hz, mag_db : output of frequency_response().
    lowcut_hz, highcut_hz : passband edges (Hz).

    Returns
    -------
    float  Peak-to-peak passband ripple in dB. Lower is better.
    """
    mask = (freqs_hz >= lowcut_hz) & (freqs_hz <= highcut_hz)
    pb = mag_db[mask]
    if len(pb) == 0:
        return float("nan")
    return float(np.max(pb) - np.min(pb))


def stopband_attenuation_db(
    freqs_hz: np.ndarray,
    mag_db: np.ndarray,
    stopband_freq_hz: float,
    side: str = "low",
) -> float:
    """
    Return attenuation (dB, positive = attenuated) at a stopband frequency.

    Quality gate: attenuation > 20 dB at DC (side='low') and at Nyquist (side='high').

    Parameters
    ----------
    stopband_freq_hz : float  Frequency to evaluate (Hz).
    side             : str    'low' (below passband) or 'high' (above passband).

    Returns
    -------
    float  Attenuation in dB (positive = signal reduced). Higher is better.
    """
    idx = np.argmin(np.abs(freqs_hz - stopband_freq_hz))
    return float(-mag_db[idx])


# ── Comparison runner ─────────────────────────────────────────────────────────

def compare_configs(
    signal_noisy: np.ndarray,
    signal_clean: np.ndarray,
    true_peak_indices: np.ndarray,
    fs: int,
    configs: list[dict],
) -> list[dict]:
    """
    Apply multiple bandpass configurations and report metrics for each.

    Per Algorithm Engineering Discipline Rule 4: reports BOTH peak-timing MAE
    AND waveform RMSE at peaks — not just HR MAE. HR MAE can look fine while
    RMSSD is already corrupted by peak-timing errors from filter ringing.

    Parameters
    ----------
    signal_noisy      : np.ndarray  Signal with noise + baseline (no ground truth).
    signal_clean      : np.ndarray  Ground-truth clean PPG.
    true_peak_indices : np.ndarray  Ground-truth systolic peak indices.
    fs                : int         Sampling rate (Hz).
    configs           : list[dict]  Each dict: {name, lowcut_hz, highcut_hz, order}.

    Returns
    -------
    list[dict]  One result dict per config with keys:
                name, lowcut_hz, highcut_hz, order,
                passband_ripple_db, stopband_atten_low_db, stopband_atten_high_db,
                residual_rmse, peak_timing_mae_ms, waveform_rmse.
    """
    n = len(signal_noisy)
    half_win = int(0.05 * fs)  # ±50 ms search window for peak detection

    if len(true_peak_indices) > 1:
        mean_rr_samples = float(np.mean(np.diff(true_peak_indices)))
        sigma_samples = int(round(0.04 * mean_rr_samples))
    else:
        sigma_samples = int(0.034 * fs)

    results = []

    for cfg in configs:
        name        = cfg["name"]
        lowcut_hz   = cfg["lowcut_hz"]
        highcut_hz  = cfg["highcut_hz"]
        order       = cfg["order"]

        result = {
            "name": name,
            "lowcut_hz": lowcut_hz,
            "highcut_hz": highcut_hz,
            "order": order,
        }

        try:
            filtered = bandpass_butterworth(
                signal_noisy, fs,
                lowcut_hz=lowcut_hz,
                highcut_hz=highcut_hz,
                order=order,
                zero_phase=True,
            )
        except Exception as e:
            result["error"] = str(e)
            results.append(result)
            continue

        # ── Frequency response metrics ─────────────────────────────────────
        freqs, mag_db = frequency_response(fs, lowcut_hz, highcut_hz, order)
        result["passband_ripple_db"]      = passband_ripple_db(freqs, mag_db, lowcut_hz, highcut_hz)
        result["stopband_atten_low_db"]   = stopband_attenuation_db(freqs, mag_db, 0.1, side="low")
        result["stopband_atten_high_db"]  = stopband_attenuation_db(freqs, mag_db, highcut_hz * 2, side="high")

        # ── Residual RMSE (whole signal) ───────────────────────────────────
        result["residual_rmse"] = float(
            np.sqrt(np.mean((filtered - signal_clean) ** 2))
        )

        # ── Peak-timing MAE (most critical for RMSSD) ─────────────────────
        timing_errors_ms = []
        for idx in true_peak_indices:
            lo = max(0, idx - half_win)
            hi = min(n, idx + half_win + 1)
            detected_idx = lo + np.argmax(filtered[lo:hi])
            timing_errors_ms.append(abs(detected_idx - idx) / fs * 1000.0)
        result["peak_timing_mae_ms"] = float(np.mean(timing_errors_ms)) if timing_errors_ms else float("nan")

        # ── Waveform RMSE at peak regions ──────────────────────────────────
        peak_samples = []
        for idx in true_peak_indices:
            lo = max(0, idx - 2 * sigma_samples)
            hi = min(n, idx + 2 * sigma_samples + 1)
            peak_samples.extend(range(lo, hi))
        peak_samples = np.unique(peak_samples)
        if len(peak_samples) > 0:
            result["waveform_rmse"] = float(
                np.sqrt(np.mean((filtered[peak_samples] - signal_clean[peak_samples]) ** 2))
            )
        else:
            result["waveform_rmse"] = float("nan")

        results.append(result)

    return results
