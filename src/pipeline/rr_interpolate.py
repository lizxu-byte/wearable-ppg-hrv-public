"""
rr_interpolate.py — RR interval resampling to uniform time grid

HRV frequency-domain analysis (Welch, AR-Burg) requires a uniformly sampled
time series. RR intervals from peak detection are non-uniformly spaced. This
module converts the irregular sequence to a 4 Hz uniform grid.

Convention: RR[n] is assigned timestamp t[n] = cumulative sum of RR[0..n-1]
(time of the n-th R-peak). This is the standard in HRV literature.

Error propagation to HRV frequency domain (downstream P1-9):
    Interpolation RMSE → spectral leakage → LF/HF power error
    Overshoot at peaks → false high-frequency energy → inflated HF band
    A cubic spline that overshoots can add ~10% spurious HF power.

Three methods implemented:
    A. cubic_spline — scipy CubicSpline (not-a-knot boundary, no extrapolation)
    B. quadratic    — scipy interp1d kind='quadratic'
    C. linear       — scipy interp1d kind='linear'

All methods clip output to physiological range [300, 2000] ms to prevent
non-physiological extrapolation at sequence boundaries.

References:
    Task Force of the European Society of Cardiology (1996). Heart rate
    variability. Circulation 93:1043-1065.
    https://doi.org/10.1161/01.CIR.93.5.1043
    (Section: Frequency-domain methods; recommends resampling to uniform
     grid before spectral analysis; 4 Hz cited as standard for HRV)

    Electrophysiology (1996). Task Force. Eur Heart J 17:354-381.
    (Same recommendation; cubic spline preferred for smoothness)

    Lomb NR (1976). Least-squares frequency analysis of unequally spaced data.
    Astrophys Space Sci 39:447-462.
    (Alternative: Lomb-Scargle avoids interpolation entirely; see P1-9)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


# ── Timestamp builder ─────────────────────────────────────────────────────────

def rr_to_timestamps(rr_ms: np.ndarray) -> np.ndarray:
    """
    Convert RR intervals to R-peak timestamps.

    Convention: t[0] = 0, t[n] = RR[0] + RR[1] + ... + RR[n-1]  (in seconds).
    The last element gives the time of the final R-peak.

    Parameters
    ----------
    rr_ms : np.ndarray  RR intervals in ms, shape (N,).

    Returns
    -------
    np.ndarray  Timestamps in seconds, shape (N,). t[0] = 0.
    """
    return np.concatenate([[0.0], np.cumsum(rr_ms[:-1])]) / 1000.0


# ── Method A: Cubic spline ────────────────────────────────────────────────────

def interpolate_cubic(
    rr_ms: np.ndarray,
    timestamps_s: np.ndarray,
    target_fs: float = 4.0,
    clip_min_ms: float = 300.0,
    clip_max_ms: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample RR intervals to a uniform grid using cubic spline interpolation.

    Uses scipy CubicSpline with 'not-a-knot' boundary condition — the most
    common choice for HRV; it minimises the third derivative at boundaries.
    No extrapolation: the output grid is clipped to [t[0], t[-1]].

    Reference: Task Force ESC (1996) Circulation 93:1043-1065.
               CubicSpline not-a-knot: de Boor (1978) A Practical Guide
               to Splines. Springer.

    Parameters
    ----------
    rr_ms        : np.ndarray  Clean RR intervals (ms), shape (N,).
    timestamps_s : np.ndarray  R-peak timestamps (s), shape (N,). Use rr_to_timestamps().
    target_fs    : float       Output sampling rate (Hz). Default 4 Hz (Task Force 1996).
    clip_min_ms  : float       Lower clip bound (ms). Default 300 ms.
    clip_max_ms  : float       Upper clip bound (ms). Default 2000 ms.
                               If sensor population changes (e.g. elderly with bradycardia):
                               raise upper bound to 2500 ms.

    Returns
    -------
    rr_uniform  : np.ndarray  Resampled RR values (ms) at uniform grid.
    t_uniform   : np.ndarray  Uniform time axis (s).
    """
    rr_ms       = np.asarray(rr_ms, dtype=float)
    timestamps_s = np.asarray(timestamps_s, dtype=float)

    t_uniform = np.arange(timestamps_s[0], timestamps_s[-1], 1.0 / target_fs)
    cs = CubicSpline(timestamps_s, rr_ms, bc_type="not-a-knot", extrapolate=False)
    rr_uniform = cs(t_uniform)

    # Fill NaN at boundaries (extrapolate=False → NaN outside range) with edge values
    rr_uniform = _fill_boundary_nan(rr_uniform, rr_ms)
    rr_uniform = np.clip(rr_uniform, clip_min_ms, clip_max_ms)
    return rr_uniform, t_uniform


# ── Method B: Quadratic interpolation ─────────────────────────────────────────

def interpolate_quadratic(
    rr_ms: np.ndarray,
    timestamps_s: np.ndarray,
    target_fs: float = 4.0,
    clip_min_ms: float = 300.0,
    clip_max_ms: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample RR intervals to a uniform grid using quadratic interpolation.

    Lower-order than cubic — less overshoot risk, but less smooth. Suitable
    when the RR sequence contains abrupt transitions (e.g. after ectopic removal).

    Parameters
    ----------
    rr_ms, timestamps_s, target_fs : same as interpolate_cubic.
    clip_min_ms, clip_max_ms       : physiological bounds (ms).

    Returns
    -------
    rr_uniform  : np.ndarray  Resampled RR values (ms).
    t_uniform   : np.ndarray  Uniform time axis (s).
    """
    rr_ms       = np.asarray(rr_ms, dtype=float)
    timestamps_s = np.asarray(timestamps_s, dtype=float)

    t_uniform = np.arange(timestamps_s[0], timestamps_s[-1], 1.0 / target_fs)
    f = interp1d(
        timestamps_s, rr_ms,
        kind="quadratic",
        bounds_error=False,
        fill_value=(rr_ms[0], rr_ms[-1]),
    )
    rr_uniform = f(t_uniform)
    rr_uniform = np.clip(rr_uniform, clip_min_ms, clip_max_ms)
    return rr_uniform, t_uniform


# ── Method C: Linear interpolation ────────────────────────────────────────────

def interpolate_linear(
    rr_ms: np.ndarray,
    timestamps_s: np.ndarray,
    target_fs: float = 4.0,
    clip_min_ms: float = 300.0,
    clip_max_ms: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample RR intervals to a uniform grid using linear interpolation.

    Simplest method — no overshoot, but introduces piecewise-linear artifacts
    that add high-frequency energy to the spectrum. This elevates the HF band
    estimate and suppresses LF/HF ratio.

    Parameters
    ----------
    rr_ms, timestamps_s, target_fs : same as interpolate_cubic.
    clip_min_ms, clip_max_ms       : physiological bounds (ms).

    Returns
    -------
    rr_uniform  : np.ndarray  Resampled RR values (ms).
    t_uniform   : np.ndarray  Uniform time axis (s).
    """
    rr_ms       = np.asarray(rr_ms, dtype=float)
    timestamps_s = np.asarray(timestamps_s, dtype=float)

    t_uniform = np.arange(timestamps_s[0], timestamps_s[-1], 1.0 / target_fs)
    f = interp1d(
        timestamps_s, rr_ms,
        kind="linear",
        bounds_error=False,
        fill_value=(rr_ms[0], rr_ms[-1]),
    )
    rr_uniform = f(t_uniform)
    rr_uniform = np.clip(rr_uniform, clip_min_ms, clip_max_ms)
    return rr_uniform, t_uniform


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fill_boundary_nan(arr: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Fill leading/trailing NaN produced by CubicSpline extrapolate=False."""
    arr = arr.copy()
    if len(arr) == 0:
        return arr
    # Leading NaN
    first_valid = 0
    while first_valid < len(arr) and np.isnan(arr[first_valid]):
        first_valid += 1
    if first_valid > 0 and first_valid < len(arr):
        arr[:first_valid] = arr[first_valid]
    elif first_valid == len(arr):
        arr[:] = original[0]
        return arr
    # Trailing NaN
    last_valid = len(arr) - 1
    while last_valid >= 0 and np.isnan(arr[last_valid]):
        last_valid -= 1
    if last_valid < len(arr) - 1:
        arr[last_valid + 1:] = arr[last_valid]
    return arr


# ── Dispatcher ────────────────────────────────────────────────────────────────

_METHODS = {
    "cubic_spline": interpolate_cubic,
    "quadratic":    interpolate_quadratic,
    "linear":       interpolate_linear,
}


def interpolate_rr(
    rr_ms: np.ndarray,
    timestamps_s: np.ndarray,
    method: str,
    target_fs: float = 4.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch to the selected RR interpolation method.

    Parameters
    ----------
    rr_ms        : np.ndarray  Clean RR intervals (ms).
    timestamps_s : np.ndarray  R-peak timestamps (s). Use rr_to_timestamps().
    method       : str         One of "cubic_spline", "quadratic", "linear".
    target_fs    : float       Output sampling rate (Hz). Default 4.
    **kwargs                   Forwarded to the method function.

    Returns
    -------
    rr_uniform : np.ndarray  Uniform-grid RR (ms).
    t_uniform  : np.ndarray  Time axis (s).
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown interpolation method '{method}'. Choose from {list(_METHODS)}")
    return _METHODS[method](rr_ms, timestamps_s, target_fs=target_fs, **kwargs)


# ── Evaluation metrics ────────────────────────────────────────────────────────

def evaluate(
    rr_uniform: np.ndarray,
    t_uniform: np.ndarray,
    rr_ms_true: np.ndarray,
    timestamps_s_true: np.ndarray,
    clip_min_ms: float = 300.0,
    clip_max_ms: float = 2000.0,
) -> dict:
    """
    Evaluate interpolation quality by re-sampling the uniform grid back to
    original timestamps and comparing with ground truth.

    Parameters
    ----------
    rr_uniform        : np.ndarray  Uniform-grid RR (ms) from interpolate_rr().
    t_uniform         : np.ndarray  Uniform time axis (s).
    rr_ms_true        : np.ndarray  Ground-truth RR values (ms).
    timestamps_s_true : np.ndarray  Ground-truth timestamps (s).
    clip_min_ms       : float       Lower physiological bound (ms).
    clip_max_ms       : float       Upper physiological bound (ms).

    Returns
    -------
    dict with keys:
        n_uniform          — length of uniform output
        rmse_ms            — RMSE vs ground truth at original timestamps
        max_error_ms       — max |error| at any original timestamp
        n_nonphysiological — count of values outside [clip_min, clip_max]
        rr_min_ms          — min value in uniform output
        rr_max_ms          — max value in uniform output
        mean_ms            — mean of uniform output
    """
    # Re-evaluate at original timestamps via linear interpolation of the uniform grid
    f_eval = interp1d(
        t_uniform, rr_uniform,
        kind="linear",
        bounds_error=False,
        fill_value=(rr_uniform[0], rr_uniform[-1]),
    )
    rr_at_orig = f_eval(timestamps_s_true)

    errors = rr_at_orig - rr_ms_true.astype(float)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_err = float(np.max(np.abs(errors)))

    n_nonphys = int(np.sum((rr_uniform < clip_min_ms) | (rr_uniform > clip_max_ms)))

    return {
        "n_uniform":          len(rr_uniform),
        "rmse_ms":            rmse,
        "max_error_ms":       max_err,
        "n_nonphysiological": n_nonphys,
        "rr_min_ms":          float(np.min(rr_uniform)),
        "rr_max_ms":          float(np.max(rr_uniform)),
        "mean_ms":            float(np.mean(rr_uniform)),
    }


# ── Comparison runner ─────────────────────────────────────────────────────────

def compare_methods(
    rr_ms: np.ndarray,
    timestamps_s: np.ndarray,
    configs: list[dict],
    target_fs: float = 4.0,
    clip_min_ms: float = 300.0,
    clip_max_ms: float = 2000.0,
) -> list[dict]:
    """
    Run multiple interpolation methods and evaluate each.

    Parameters
    ----------
    rr_ms        : np.ndarray  Clean RR intervals (ms).
    timestamps_s : np.ndarray  R-peak timestamps (s).
    configs      : list[dict]  Each: {name, method, **kwargs}.
    target_fs    : float       Uniform output rate (Hz). Default 4.
    clip_min_ms  : float       Lower physiological bound (ms).
    clip_max_ms  : float       Upper physiological bound (ms).

    Returns
    -------
    list[dict]  Metrics per method.
    """
    results = []
    for cfg in configs:
        name   = cfg["name"]
        method = cfg["method"]
        kwargs = {k: v for k, v in cfg.items() if k not in ("name", "method")}
        row = {"name": name, "method": method}
        try:
            rr_uni, t_uni = interpolate_rr(
                rr_ms, timestamps_s, method, target_fs=target_fs, **kwargs
            )
            row.update(evaluate(
                rr_uni, t_uni, rr_ms, timestamps_s,
                clip_min_ms=clip_min_ms, clip_max_ms=clip_max_ms,
            ))
        except Exception as e:
            row["error"] = str(e)
        results.append(row)
    return results
