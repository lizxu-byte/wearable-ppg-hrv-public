"""
hrv_freq.py — HRV frequency-domain spectral methods

Three spectral estimation methods compared:
    A. Welch        — non-parametric, Hann window, on interpolated 4 Hz RR
    B. AR-Burg      — parametric, order 16, on interpolated 4 Hz RR
    C. Lomb-Scargle — non-uniform, directly on original non-uniform RR

Frequency bands (Task Force 1996):
    VLF : 0.003 – 0.04 Hz  (very low frequency; requires ≥5 min)
    LF  : 0.04  – 0.15 Hz  (low frequency; sympathetic + parasympathetic)
    HF  : 0.15  – 0.40 Hz  (high frequency; parasympathetic / respiratory)

Error propagation note (Algorithm Engineering Discipline Rule 2):
    Interpolation → spectral leakage → LF/HF ratio error.
    AR model order → over-/under-smoothing → peak position shift.
    These propagate directly to LF/HF ratio, the key autonomic metric.

References:
    A+B. Task Force ESC/NASPE (1996). Heart rate variability. Circulation
         93:1043-1065.
         https://doi.org/10.1161/01.CIR.93.5.1043
         (VLF/LF/HF band definitions; recommends Welch and AR methods)

    B.   Burg JP (1967). Maximum entropy spectral analysis. Proc 37th
         Meeting Society Exploration Geophysicists.
         Marple SL (1987). Digital Spectral Analysis with Applications.
         Prentice-Hall. (Chapter 8: Burg algorithm implementation)

    C.   Lomb NR (1976). Least-squares frequency analysis of unequally
         spaced data. Astrophys Space Sci 39:447-462.
         Scargle JD (1982). Studies in astronomical time series. II.
         Astrophys J 263:835-853.
         VanderPlas JT (2018). Understanding the Lomb-Scargle Periodogram.
         Astrophys J Suppl Ser 236:16.
         https://doi.org/10.3847/1538-4365/aab766
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch, lombscargle


# ── Band power helper ─────────────────────────────────────────────────────────

def band_power(freqs: np.ndarray, psd: np.ndarray, low_hz: float, high_hz: float) -> float:
    """
    Integrate PSD over a frequency band using the trapezoidal rule.

    Parameters
    ----------
    freqs   : np.ndarray  Frequency axis (Hz).
    psd     : np.ndarray  Power spectral density (ms²/Hz).
    low_hz  : float       Lower band edge (Hz).
    high_hz : float       Upper band edge (Hz).

    Returns
    -------
    float  Band power in ms².
    """
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if np.sum(mask) < 2:
        return float("nan")
    return float(np.trapezoid(psd[mask], freqs[mask]))


def compute_hrv_freq(freqs: np.ndarray, psd: np.ndarray, bands: dict) -> dict:
    """
    Extract band powers and derived HRV metrics from a PSD estimate.

    Parameters
    ----------
    freqs : np.ndarray  Frequency axis (Hz).
    psd   : np.ndarray  One-sided PSD (ms²/Hz).
    bands : dict        Band definitions: {"vlf": (lo, hi), "lf": (lo, hi), "hf": (lo, hi)}.

    Returns
    -------
    dict with keys:
        vlf_ms2, lf_ms2, hf_ms2, total_ms2,
        lf_norm_pct, hf_norm_pct, lf_hf_ratio
    """
    vlf = band_power(freqs, psd, *bands["vlf"])
    lf  = band_power(freqs, psd, *bands["lf"])
    hf  = band_power(freqs, psd, *bands["hf"])
    total = band_power(freqs, psd, bands["vlf"][0], bands["hf"][1])

    lf_hf   = lf / hf if (np.isfinite(hf) and hf > 0) else float("nan")
    lf_norm = lf / (lf + hf) * 100 if (np.isfinite(lf + hf) and (lf + hf) > 0) else float("nan")
    hf_norm = hf / (lf + hf) * 100 if (np.isfinite(lf + hf) and (lf + hf) > 0) else float("nan")

    return {
        "vlf_ms2":     vlf,
        "lf_ms2":      lf,
        "hf_ms2":      hf,
        "total_ms2":   total,
        "lf_hf_ratio": lf_hf,
        "lf_norm_pct": lf_norm,
        "hf_norm_pct": hf_norm,
    }


# ── Method A: Welch ───────────────────────────────────────────────────────────

def welch_psd(
    rr_uniform_ms: np.ndarray,
    fs: float = 4.0,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate PSD of the uniform RR series using Welch's method.

    Uses Hann window (scipy default). Input should be the interpolated
    uniform-grid RR sequence from rr_interpolate.py.

    Reference: Task Force ESC (1996) Circulation 93:1043-1065.
               Welch PD (1967). The use of fast Fourier transform for the
               estimation of power spectra. IEEE Trans Audio Electroacoustics
               15(2):70-73.

    Parameters
    ----------
    rr_uniform_ms : np.ndarray  Uniform-grid RR (ms), 1-D.
    fs            : float       Sampling rate (Hz). Default 4.
    nperseg       : int         Segment length (samples). Default 256.
                                At 4 Hz: 256 → 64 s segments → freq resolution 0.016 Hz.
                                For VLF analysis: increase to 512 (128 s).
    noverlap      : int | None  Overlap samples. Default None → nperseg // 2.

    Returns
    -------
    freqs_hz : np.ndarray  Frequency axis (Hz).
    psd      : np.ndarray  One-sided PSD (ms²/Hz).
    """
    rr = np.asarray(rr_uniform_ms, dtype=float)
    rr_detrended = rr - np.mean(rr)
    freqs, psd = welch(
        rr_detrended,
        fs=fs,
        window="hann",
        nperseg=min(nperseg, len(rr_detrended)),
        noverlap=noverlap,
        scaling="density",
    )
    return freqs, psd


# ── Method B: AR-Burg ─────────────────────────────────────────────────────────

def _burg_ar(x: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """
    Estimate AR coefficients using the Burg algorithm.

    Reference: Burg JP (1967); Marple (1987) Chapter 8.

    Parameters
    ----------
    x     : np.ndarray  Zero-mean input signal, 1-D.
    order : int         AR model order.

    Returns
    -------
    ar       : np.ndarray  AR coefficients [a(1), ..., a(p)], length order.
    noise_var: float       Residual process variance (ms²).
    """
    N = len(x)
    x = (x - np.mean(x)).astype(float)

    f = x.copy()   # forward prediction error
    b = x.copy()   # backward prediction error

    ar = np.zeros(order, dtype=float)
    noise_var = float(np.dot(x, x) / N)

    for m in range(1, order + 1):
        # Valid slice: f[m:] and b[m-1:N-1], both length N-m
        ef = f[m:].copy()
        eb = b[m - 1: N - 1].copy()

        num = -2.0 * np.dot(ef, eb)
        den = np.dot(ef, ef) + np.dot(eb, eb)
        km  = (num / den) if abs(den) > 1e-12 else 0.0

        # Update prediction errors
        f[m:]      = ef + km * eb
        b[m - 1: N - 1] = eb + km * ef  # uses old ef above

        # Levinson recursion: update AR coefficients
        a_prev = ar[: m - 1].copy()
        for j in range(m - 1):
            ar[j] = a_prev[j] + km * a_prev[m - 2 - j]
        ar[m - 1] = km

        noise_var *= max(1.0 - km ** 2, 1e-12)

    return ar, noise_var


def ar_burg_psd(
    rr_uniform_ms: np.ndarray,
    fs: float = 4.0,
    order: int = 16,
    n_fft: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate PSD using the Burg AR parametric method.

    Advantages over Welch: higher frequency resolution for short recordings;
    smooth spectral estimates without windowing artifacts.
    Disadvantages: frequency peaks can shift if order is misspecified.

    Reference: Task Force ESC (1996) Circulation 93:1043-1065.
               (Recommends AR method for short-term HRV recordings)
               Order 16 per Burg (1967); also used in nk.hrv_frequency.

    Parameters
    ----------
    rr_uniform_ms : np.ndarray  Uniform-grid RR (ms), 1-D.
    fs            : float       Sampling rate (Hz). Default 4.
    order         : int         AR model order. Default 16.
                                Task Force 1996: order 8–20 for 5-min HRV.
                                For longer recordings: higher order can be used.
    n_fft         : int         Number of frequency points in output PSD.

    Returns
    -------
    freqs_hz : np.ndarray  Frequency axis (Hz), 0 to fs/2.
    psd      : np.ndarray  One-sided PSD (ms²/Hz).
    """
    rr = np.asarray(rr_uniform_ms, dtype=float)
    ar_coef, noise_var = _burg_ar(rr, order)

    # Evaluate PSD on frequency grid
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # A(f) = 1 + a[0]*e^{-j2πf/fs} + a[1]*e^{-j2πf*2/fs} + ...
    k = np.arange(1, order + 1, dtype=float)
    exp_mat = np.exp(-1j * 2 * np.pi * np.outer(freqs, k) / fs)
    A = 1.0 + (ar_coef[np.newaxis, :] * exp_mat).sum(axis=1)

    # Two-sided PSD: σ²/fs / |A|²
    psd_2s = noise_var / fs / (np.abs(A) ** 2)

    # Convert to one-sided: double all bins except DC and Nyquist
    psd_1s = 2.0 * psd_2s
    psd_1s[0] = psd_2s[0]          # DC: no doubling
    if len(freqs) > 1 and n_fft % 2 == 0:
        psd_1s[-1] = psd_2s[-1]    # Nyquist: no doubling

    return freqs, psd_1s


# ── Method C: Lomb-Scargle ────────────────────────────────────────────────────

def lomb_scargle_psd(
    rr_ms: np.ndarray,
    timestamps_s: np.ndarray,
    freq_min_hz: float = 0.003,
    freq_max_hz: float = 0.5,
    n_freqs: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate PSD using the Lomb-Scargle periodogram for non-uniform data.

    Unlike Welch and AR, Lomb-Scargle operates directly on the original
    (non-interpolated) RR intervals, avoiding interpolation-induced distortion.
    This is the preferred method when the interpolation quality is uncertain.

    Normalization: output PSD is scaled so that the integral over all
    frequencies equals the variance of the input signal (ms²).

    References: Lomb (1976); Scargle (1982); VanderPlas (2018).

    Parameters
    ----------
    rr_ms        : np.ndarray  Clean RR intervals (ms), 1-D.
    timestamps_s : np.ndarray  R-peak timestamps (s), from rr_to_timestamps().
    freq_min_hz  : float       Lowest frequency to evaluate (Hz). Default 0.003.
    freq_max_hz  : float       Highest frequency (Hz). Default 0.5.
    n_freqs      : int         Number of frequency points.

    Returns
    -------
    freqs_hz : np.ndarray  Frequency axis (Hz).
    psd      : np.ndarray  PSD (ms²/Hz).
    """
    rr = np.asarray(rr_ms, dtype=float)
    t  = np.asarray(timestamps_s, dtype=float)
    rr_detrended = rr - np.mean(rr)

    freqs_hz  = np.linspace(freq_min_hz, freq_max_hz, n_freqs)
    ang_freqs = 2 * np.pi * freqs_hz

    # scipy lombscargle: normalize=False → pgram in units of (rr_detrended)²
    pgram = lombscargle(t, rr_detrended, ang_freqs, normalize=False)

    # Convert to PSD in ms²/Hz
    # Scargle (1982) normalization: PSD = pgram × 4 / (N × df)
    # where N = len(rr) and df = freq spacing
    N  = len(rr)
    df = freqs_hz[1] - freqs_hz[0]
    psd = pgram * 4.0 / (N * df)

    return freqs_hz, psd


# ── Dispatcher ────────────────────────────────────────────────────────────────

_METHODS = {
    "welch":         welch_psd,
    "ar_burg":       ar_burg_psd,
    "lomb_scargle":  lomb_scargle_psd,
}


def compute_psd(
    method: str,
    rr_uniform_ms: np.ndarray | None = None,
    rr_ms: np.ndarray | None = None,
    timestamps_s: np.ndarray | None = None,
    fs: float = 4.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch to the selected PSD estimation method.

    Parameters
    ----------
    method          : str  One of "welch", "ar_burg", "lomb_scargle".
    rr_uniform_ms   : np.ndarray  Required for "welch" and "ar_burg".
    rr_ms           : np.ndarray  Required for "lomb_scargle".
    timestamps_s    : np.ndarray  Required for "lomb_scargle".
    fs              : float       Sampling rate (Hz). Used by welch/ar_burg.
    **kwargs                      Forwarded to the method function.

    Returns
    -------
    freqs : np.ndarray  Frequency axis (Hz).
    psd   : np.ndarray  PSD (ms²/Hz).
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown spectral method '{method}'. Choose from {list(_METHODS)}")

    if method in ("welch", "ar_burg"):
        if rr_uniform_ms is None:
            raise ValueError(f"'{method}' requires rr_uniform_ms")
        return _METHODS[method](rr_uniform_ms, fs=fs, **kwargs)
    else:
        if rr_ms is None or timestamps_s is None:
            raise ValueError("'lomb_scargle' requires rr_ms and timestamps_s")
        return _METHODS[method](rr_ms, timestamps_s, **kwargs)
