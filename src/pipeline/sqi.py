"""
sqi.py — Signal Quality Index for PPG windows

Two complementary quality measures:

    A. Template Correlation SQI (tc_sqi)
       Measures morphological consistency across beats.
       Each beat is resampled and correlated with the session median template.
       Uniform, repeating beats → high correlation → high quality.

    B. Spectral Entropy SQI (se_sqi)
       Measures frequency content concentration in the physiological band.
       Clean PPG has power concentrated at HR harmonics → low entropy → high quality.
       White noise / motion artifact spreads power → high entropy → low quality.

Both return scores in [0, 1] where higher = better quality.
NaN is returned when insufficient data exists to compute the measure.

Error propagation (Algorithm Engineering Discipline Rule 2):
    Low-SQI windows excluded from HRV pipeline → prevents RMSSD contamination.
    Template correlation targets beat-level artifact (ectopic, motion transient).
    Spectral entropy targets broad-band noise and sustained motion artifact.

References:
    Orphanidou C et al. (2015). Signal-Quality Indices for the Electrocardiogram
    and Photoplethysmogram. IEEE J Biomed Health Inform 19(3):832-838.
    https://doi.org/10.1109/JBHI.2014.2338351
    (Template correlation SQI for PPG; threshold sqi_t ≥ 0.66)

    Li Q et al. (2012). Robust heart rate estimation from multiple asynchronous
    noisy channels. J Clin Monit Comput 22:137-145.
    https://doi.org/10.1007/s10877-008-9132-9
    (Spectral entropy for multi-channel PPG quality; used in PhysioNet challenge)

    Shannon CE (1948). A mathematical theory of communication.
    Bell System Technical Journal 27:379-423.
    (Shannon entropy; used here on normalized PSD to quantify spectral spread)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import periodogram

from src.pipeline.peaks import detect_peaks


# ── Method A: Template Correlation SQI ────────────────────────────────────────

def template_correlation_sqi(
    signal: np.ndarray,
    fs: float,
    n_template_points: int = 100,
    min_beats: int = 5,
) -> float:
    """
    Template correlation SQI — beat morphology consistency.

    Detects peaks, extracts each interbeat segment, resamples to
    n_template_points, and correlates with the median template.
    Beats with physiologically implausible lengths are skipped.

    Reference: Orphanidou et al. (2015) IEEE J Biomed Health Inform 19(3):832-838.

    Parameters
    ----------
    signal           : np.ndarray  Filtered PPG window (mV or a.u.).
    fs               : float       Sampling rate (Hz). Typically 256.
    n_template_points: int         Template length (samples). Default 100.
    min_beats        : int         Minimum valid beats; returns NaN if fewer.
                                   Minimum 5 for a stable median template.
                                   For HRV work (60–100 bpm, 10s window):
                                   expect 10–17 beats → well above minimum.
                                   Increase for population with arrhythmia.

    Returns
    -------
    float  Mean Pearson r with median template, clipped to [0, 1].
           NaN if fewer than min_beats physiologically valid beats found.
    """
    signal = np.asarray(signal, dtype=float)
    peaks = detect_peaks(signal, fs, method="prominence", dedup=True)

    if len(peaks) < min_beats + 1:
        return float("nan")

    beats: list[np.ndarray] = []
    for i in range(len(peaks) - 1):
        n_samples = peaks[i + 1] - peaks[i]
        duration_ms = n_samples / fs * 1000.0
        if not (300.0 <= duration_ms <= 2000.0):
            continue
        beat = signal[peaks[i] : peaks[i + 1]]
        x_orig = np.linspace(0.0, 1.0, len(beat))
        x_new  = np.linspace(0.0, 1.0, n_template_points)
        beat_r = np.interp(x_new, x_orig, beat)
        std = float(np.std(beat_r))
        if std < 1e-10:
            continue
        beats.append((beat_r - np.mean(beat_r)) / std)

    if len(beats) < min_beats:
        return float("nan")

    mat = np.array(beats)                                # (n_beats, n_template)
    template = np.median(mat, axis=0)
    std_t = float(np.std(template))
    if std_t < 1e-10:
        return float("nan")
    template = (template - np.mean(template)) / std_t

    correlations = np.array(
        [float(np.corrcoef(b, template)[0, 1]) for b in mat]
    )
    # Clip: negative correlation = inverted beat = artifact → treat as 0
    return float(np.mean(np.clip(correlations, 0.0, 1.0)))


# ── Method B: Spectral Entropy SQI ────────────────────────────────────────────

def spectral_entropy_sqi(
    signal: np.ndarray,
    fs: float,
    band_hz: tuple[float, float] = (0.5, 8.0),
) -> float:
    """
    Spectral entropy SQI — frequency-domain quality measure.

    Computes the normalized Shannon entropy of the power spectral density
    within the physiological PPG band. Lower entropy (power concentrated
    at heartbeat harmonics) → higher quality.

    Returns 1 − (H / H_max) so that output is always:
        0 = maximal entropy (white noise, uniform spectrum)
        1 = minimal entropy (pure sinusoid at one frequency)

    Reference: Li et al. (2012) J Clin Monit Comput 22:137-145.

    Parameters
    ----------
    signal  : np.ndarray   PPG signal window (mV or a.u.).
    fs      : float        Sampling rate (Hz). Typically 256.
    band_hz : tuple        Physiological band (Hz). Default (0.5, 8.0).
                           Covers HR range 30–480 bpm including harmonics.
                           PPG fingertip: adjust upper edge if no harmonics
                           above 5 Hz are needed (lower HR population).

    Returns
    -------
    float  SQI in [0, 1]; higher = better quality.
           Returns NaN if fewer than 3 frequency bins in band.
    """
    signal = np.asarray(signal, dtype=float)
    freqs, psd = periodogram(signal - np.mean(signal), fs=fs)

    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if np.sum(mask) < 3:
        return float("nan")

    psd_band = psd[mask]
    total = float(np.sum(psd_band))
    if total < 1e-30:
        return float("nan")

    p = psd_band / total                          # normalised power distribution
    # Guard against zero entries before log (unused bins should not appear in
    # band after the mask, but float rounding can produce tiny negatives)
    p = np.clip(p, 1e-12, None)

    h = -float(np.sum(p * np.log2(p)))           # Shannon entropy (bits)
    h_max = np.log2(float(np.sum(mask)))          # entropy of uniform distribution

    if h_max < 1e-10:
        return float("nan")

    return float(1.0 - h / h_max)


# ── Combined dispatcher ────────────────────────────────────────────────────────

def compute_sqi(
    signal: np.ndarray,
    fs: float,
    methods: tuple[str, ...] = ("template_correlation", "spectral_entropy"),
    **kwargs,
) -> dict[str, float]:
    """
    Compute SQI using one or more methods.

    Parameters
    ----------
    signal  : np.ndarray  PPG window (mV or a.u.).
    fs      : float       Sampling rate (Hz).
    methods : tuple       Subset of {"template_correlation", "spectral_entropy"}.
    **kwargs              Forwarded to individual SQI functions.

    Returns
    -------
    dict  Keys from ``methods``, values in [0, 1] or NaN.
    """
    _dispatch = {
        "template_correlation": template_correlation_sqi,
        "spectral_entropy":     spectral_entropy_sqi,
    }
    unknown = set(methods) - set(_dispatch)
    if unknown:
        raise ValueError(f"Unknown SQI method(s): {unknown}. Choose from {list(_dispatch)}")

    return {m: _dispatch[m](signal, fs, **kwargs) for m in methods}


# ── Windowed SQI over a long signal ───────────────────────────────────────────

def windowed_sqi(
    signal: np.ndarray,
    fs: float,
    window_s: float = 10.0,
    step_s: float = 5.0,
    methods: tuple[str, ...] = ("template_correlation", "spectral_entropy"),
) -> dict[str, np.ndarray]:
    """
    Compute SQI in sliding windows over a session-length signal.

    Parameters
    ----------
    signal   : np.ndarray  Full-session PPG signal.
    fs       : float       Sampling rate (Hz).
    window_s : float       Window length (seconds). Default 10 s.
                           Minimum recommended: 10 s (to yield ≥ 10 beats).
    step_s   : float       Stride between windows (seconds). Default 5 s.
    methods  : tuple       SQI methods to compute.

    Returns
    -------
    dict with keys matching ``methods``, each mapped to a 1-D float array
    of per-window SQI scores, plus "window_centers_s" for the time axis.
    """
    n_window = int(window_s * fs)
    n_step   = int(step_s   * fs)
    n_total  = len(signal)

    results: dict[str, list[float]] = {m: [] for m in methods}
    centers: list[float] = []

    start = 0
    while start + n_window <= n_total:
        window = signal[start : start + n_window]
        sqi = compute_sqi(window, fs, methods=methods)
        for m in methods:
            results[m].append(sqi[m])
        centers.append((start + n_window / 2) / fs)
        start += n_step

    return {m: np.array(results[m]) for m in methods} | {
        "window_centers_s": np.array(centers)
    }
