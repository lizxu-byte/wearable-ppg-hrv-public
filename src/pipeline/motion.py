"""
motion.py — Motion artifact detection from accelerometer data

Uses Shimmer3 three-axis accelerometer (calibrated, m/s²) to produce
a per-sample binary motion mask. High-motion samples are flagged for
exclusion from PPG and HRV analysis.

Two complementary detection methods:

    A. SVM-RMS threshold (Salehizadeh 2014)
       Signal Vector Magnitude → high-pass to remove gravity DC →
       rolling RMS envelope → threshold.
       More accurate for sustained motion; suppresses single-sample spikes.

    B. SVM rolling-std threshold
       SVMag rolling standard deviation → threshold.
       No filtering needed; robust to brief transients.
       Slightly higher false-positive rate for low-frequency body sway.

Configuration parameters (all from config.yaml, none hardcoded):
    motion.accel_magnitude_threshold  — RMS or std threshold (m/s²)
    motion.use_column                 — "wr" (wide-range) or "ln" (low-noise)

Error propagation (Algorithm Engineering Discipline Rule 2):
    motion mask → exclude PPG windows from peak detection →
    no false IBI from motion-corrupted beats →
    RMSSD / LF-HF uncontaminated.
    False negatives increase RMSSD (motion = irregular IBI).
    False positives reduce analysable recording duration.

References:
    Salehizadeh SMA et al. (2014). A novel time-varying spectral filtering
    algorithm for reconstruction of motion artifact corrupted heart rate
    signals during intense physical activities using a wearable
    photoplethysmogram sensor. Sensors 14(3):3872-3897.
    https://doi.org/10.3390/s140303872
    (SVM-based motion detection and PPG artifact removal during exercise)

    van Hees VT et al. (2013). Separating movement and gravity components
    in an acceleration signal and implications for the assessment of human
    daily physical activity. PLoS ONE 8(4):e61691.
    https://doi.org/10.1371/journal.pone.0061691
    (SVMminus gravity removal for accelerometers; used here as high-pass)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import sosfiltfilt, butter


# ── Helpers ───────────────────────────────────────────────────────────────────

def svm_magnitude(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
) -> np.ndarray:
    """
    Compute Signal Vector Magnitude: SVMag = sqrt(ax²+ay²+az²).

    For a Shimmer3 unit at rest, SVMag ≈ 9.81 m/s² (gravity on one axis).
    Motion adds energy on all axes and increases SVMag.

    Parameters
    ----------
    ax, ay, az : np.ndarray  Calibrated accelerometer axes (m/s²), equal length.

    Returns
    -------
    np.ndarray  SVMag (m/s²), same length as inputs.
    """
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    az = np.asarray(az, dtype=float)
    return np.sqrt(ax ** 2 + ay ** 2 + az ** 2)


def _rolling_rms(x: np.ndarray, n: int) -> np.ndarray:
    """
    Compute causal rolling RMS over a window of n samples using a cumsum trick.
    Output has same length as x; first n-1 samples use available data.
    """
    x2 = x ** 2
    cs = np.cumsum(np.concatenate([[0.0], x2]))
    # For each position i, window covers max(0, i-n+1) to i
    window = np.minimum(np.arange(1, len(x) + 1), n)
    return np.sqrt((cs[n:] - cs[:-n]) / n if len(x) >= n
                   else np.array([np.sqrt(np.mean(x2[:i+1])) for i in range(len(x))],
                                 dtype=float))


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    """
    Compute causal rolling std over a window of n samples.
    Output has same length as x.
    """
    out = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        segment = x[max(0, i - n + 1) : i + 1]
        out[i] = float(np.std(segment, ddof=0))
    return out


def _morph_mask(
    mask: np.ndarray,
    min_duration_samples: int,
    gap_fill_samples: int,
) -> np.ndarray:
    """
    Apply morphological open + close to clean up the motion mask:
    - Remove motion events shorter than min_duration_samples (opening).
    - Fill rest gaps shorter than gap_fill_samples within motion events (closing).

    Parameters
    ----------
    mask                 : np.ndarray  Bool/int binary mask (1=motion, 0=rest).
    min_duration_samples : int  Minimum motion event length to keep.
    gap_fill_samples     : int  Maximum rest gap to fill inside a motion event.

    Returns
    -------
    np.ndarray  Cleaned binary mask (dtype bool).
    """
    struct_open  = np.ones(min_duration_samples, dtype=bool)
    struct_close = np.ones(gap_fill_samples,     dtype=bool)
    m = mask.astype(bool)
    m = binary_opening(m, structure=struct_open)
    m = binary_closing(m, structure=struct_close)
    return m


# ── Method A: SVM-RMS threshold ───────────────────────────────────────────────

def detect_motion_threshold(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    fs: float,
    threshold_ms2: float = 2.0,
    smooth_window_s: float = 0.5,
    highpass_hz: float = 0.5,
    highpass_order: int = 4,
    min_duration_s: float = 0.5,
    gap_fill_s: float = 0.5,
) -> np.ndarray:
    """
    Detect motion via per-axis high-pass then combined SVMag rolling-RMS threshold.

    IMPORTANT: high-pass filtering is applied per axis BEFORE computing SVMag,
    not to SVMag itself. Computing HP(SVMag) underestimates motion energy because
    gravity (~9.81 m/s²) causes a nonlinear suppression of the AC amplitude in
    the magnitude domain. Per-axis HP preserves the full motion amplitude on
    each axis before combination.

    At rest:   svm_ac = sqrt(ε_x²+ε_y²+ε_z²) ≈ 0.35 m/s² (sensor noise floor)
    Walking:   svm_ac ≈ amp × sqrt(1+0.49+0.25) / √2 ≈ 4.7 m/s² for amp=5 m/s²
    Threshold 2.0 m/s²: ~6× above noise floor; detects motion ≥ 2.4 m/s² per-axis.

    Reference: Salehizadeh et al. (2014) Sensors 14(3):3872-3897.

    Parameters
    ----------
    ax, ay, az      : np.ndarray  Calibrated accelerometer (m/s²).
    fs              : float       Sampling rate (Hz). Typically 256.
    threshold_ms2   : float       RMS threshold (m/s²). Default 2.0.
                                  config.yaml: motion.accel_magnitude_threshold.
                                  At rest: svm_ac RMS ≈ 0.35 m/s² (3-axis noise).
                                  Threshold 2.0 → ~6× margin above noise floor.
                                  Reduce to 1.0 for wrist sensors (higher baseline).
    smooth_window_s : float       Rolling RMS window length (s). Default 0.5 s.
    highpass_hz     : float       Per-axis HP edge to remove gravity (Hz). Default 0.5.
                                  van Hees et al. (2013): 0.5 Hz removes gravity DC.
    highpass_order  : int         Filter order. Default 4.
    min_duration_s  : float       Minimum motion event duration (s). Default 0.5.
    gap_fill_s      : float       Rest gaps shorter than this are merged (s). Default 0.5.

    Returns
    -------
    np.ndarray  Boolean motion mask; True = motion detected. Same length as ax.
    """
    ax_a = np.asarray(ax, dtype=float)
    ay_a = np.asarray(ay, dtype=float)
    az_a = np.asarray(az, dtype=float)

    # High-pass each axis independently to remove gravity DC
    sos = butter(highpass_order, highpass_hz, btype="high", fs=fs, output="sos")
    hp_x = sosfiltfilt(sos, ax_a)
    hp_y = sosfiltfilt(sos, ay_a)
    hp_z = sosfiltfilt(sos, az_a)

    # Combined AC magnitude (no gravity component)
    svm_ac = np.sqrt(hp_x ** 2 + hp_y ** 2 + hp_z ** 2)

    # Rolling RMS envelope (offline symmetric convolution)
    n_win  = max(1, int(smooth_window_s * fs))
    kernel = np.ones(n_win) / n_win
    rms_env = np.sqrt(np.convolve(svm_ac ** 2, kernel, mode="same"))

    raw_mask = rms_env > threshold_ms2

    n_min  = max(1, int(min_duration_s * fs))
    n_fill = max(1, int(gap_fill_s     * fs))
    return _morph_mask(raw_mask, n_min, n_fill)


# ── Method B: SVM rolling-std threshold ───────────────────────────────────────

def detect_motion_variance(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    fs: float,
    threshold_ms2: float = 2.0,
    window_s: float = 0.5,
    min_duration_s: float = 0.5,
    gap_fill_s: float = 0.5,
) -> np.ndarray:
    """
    Detect motion via combined per-axis rolling variance (no HP filter required).

    Computes Var(axis) = E[x²] − E[x]² per axis, then combines:
        combined_std = sqrt(Var_x + Var_y + Var_z)

    Gravity is DC on whichever axis is vertical; rolling variance removes DC
    automatically (E[x]² subtracts the within-window mean). No explicit
    HP filter needed, making this method robust to slow body tilt.

    At rest: combined_std ≈ sqrt(3) × σ_noise ≈ 0.35 m/s² (noise floor).
    Walking: combined_std ≈ amp × sqrt(1 + 0.49 + 0.25) ≈ 4.7 m/s² for amp=5.

    Reference: van Hees et al. (2013) PLoS ONE 8(4):e61691.
               (Rolling variance as motion energy estimate for accelerometers)

    Parameters
    ----------
    ax, ay, az    : np.ndarray  Calibrated accelerometer (m/s²).
    fs            : float       Sampling rate (Hz).
    threshold_ms2 : float       Combined-std threshold (m/s²). Default 2.0.
                                Same value as Method A for direct comparability.
                                Higher combined_std → more robust to directional
                                sensitivity than single-axis std.
    window_s      : float       Rolling variance window (s). Default 0.5.
    min_duration_s: float       Minimum event duration (s). Default 0.5.
    gap_fill_s    : float       Gap fill duration (s). Default 0.5.

    Returns
    -------
    np.ndarray  Boolean motion mask; True = motion detected. Same length as ax.
    """
    n_win  = max(2, int(window_s * fs))
    kernel = np.ones(n_win) / n_win

    combined_var = np.zeros(len(ax), dtype=float)
    for arr in (ax, ay, az):
        a = np.asarray(arr, dtype=float)
        # Var(a) = E[a²] - E[a]²  (Welford/König–Huygens formula)
        e_a2 = np.convolve(a ** 2,   kernel, mode="same")
        e_a  = np.convolve(a,        kernel, mode="same")
        var_a = np.maximum(e_a2 - e_a ** 2, 0.0)   # guard float rounding
        combined_var += var_a

    combined_std = np.sqrt(combined_var)
    raw_mask = combined_std > threshold_ms2

    n_min  = max(1, int(min_duration_s * fs))
    n_fill = max(1, int(gap_fill_s     * fs))
    return _morph_mask(raw_mask, n_min, n_fill)


# ── Dispatcher ────────────────────────────────────────────────────────────────

_METHODS = {
    "threshold": detect_motion_threshold,
    "variance":  detect_motion_variance,
}


def detect_motion(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    fs: float,
    method: str = "threshold",
    **kwargs,
) -> np.ndarray:
    """
    Detect motion from three-axis accelerometer data.

    Parameters
    ----------
    ax, ay, az : np.ndarray  Calibrated accelerometer axes (m/s²).
    fs         : float       Sampling rate (Hz).
    method     : str         "threshold" (Method A, SVM-RMS) or
                             "variance"  (Method B, rolling-std).
    **kwargs                 Forwarded to the selected detection function.

    Returns
    -------
    np.ndarray  Boolean motion mask (True = motion). Same length as ax.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(_METHODS)}")
    return _METHODS[method](ax, ay, az, fs, **kwargs)


# ── Per-window summary ────────────────────────────────────────────────────────

def motion_fraction(mask: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Compute the fraction of samples in motion for each non-overlapping window.

    Parameters
    ----------
    mask            : np.ndarray  Boolean motion mask (per sample).
    window_samples  : int         Window length in samples.

    Returns
    -------
    np.ndarray  Float array of motion fraction per window, in [0, 1].
    """
    n_full = len(mask) // window_samples
    trimmed = mask[: n_full * window_samples].reshape(n_full, window_samples)
    return trimmed.mean(axis=1)
