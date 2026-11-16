"""
hrv_nonlinear.py — HRV nonlinear Poincaré metrics

The Poincaré plot visualises RR[n] on the x-axis against RR[n+1] on the
y-axis. The spread of the cloud along and perpendicular to the line of
identity characterises short-term (SD1) and long-term (SD2) variability.

Exact mathematical relationships (Brennan 2001, Equations 4–6):
    SD1² = 0.5 × Var(RR[n+1] − RR[n])          [Eq. 4]
    SD2² = 2 × Var(RR) − 0.5 × Var(RR[n+1] − RR[n])  [Eq. 5]
         = 2 × SDNN² − SD1²                      [simplified]
    SD1/SD2 → sympathovagal balance indicator
    Area = π × SD1 × SD2                          [Eq. 6]

Approximation with time-domain metrics (sample statistics, ddof=1):
    SD1 ≈ RMSSD / √2  (exact only when mean(diff_rr)=0 and N→∞; sample error O(1/N))
    SD2 ≈ √(2 × SDNN² − RMSSD²/2)  (same caveat)

References:
    Brennan M, Palaniswami M, Kamen P (2001). Do existing measures of
    Poincaré plot geometry reflect nonlinear features of heart rate
    variability? IEEE Trans Biomed Eng 48(11):1342-1347.
    https://doi.org/10.1109/10.959330

    Task Force ESC/NASPE (1996). Heart rate variability. Circulation
    93:1043-1065.
    https://doi.org/10.1161/01.CIR.93.5.1043
"""

from __future__ import annotations

import numpy as np


def compute_hrv_nonlinear(rr_ms: np.ndarray) -> dict:
    """
    Compute Poincaré plot HRV nonlinear metrics.

    Reference: Brennan et al. (2001) IEEE Trans Biomed Eng 48(11):1342-1347.

    Parameters
    ----------
    rr_ms : np.ndarray  Clean RR intervals (ms), 1-D, ≥ 3 values.

    Returns
    -------
    dict with keys:
        sd1_ms       — short-term variability (ms); reflects RMSSD / √2
        sd2_ms       — long-term variability (ms)
        sd1_sd2      — SD1/SD2 ratio (dimensionless)
        area_ms2     — Poincaré ellipse area = π × SD1 × SD2 (ms²)
        rr_n         — RR[0..N-2] for Poincaré plot x-axis
        rr_n1        — RR[1..N-1] for Poincaré plot y-axis

    Raises
    ------
    ValueError  if rr_ms has fewer than 3 elements.
    """
    rr_ms = np.asarray(rr_ms, dtype=float)
    if len(rr_ms) < 3:
        raise ValueError(f"Need ≥ 3 RR intervals; got {len(rr_ms)}")
    if np.any(rr_ms <= 0):
        raise ValueError("RR intervals must be positive")

    diff_rr = np.diff(rr_ms)      # RR[n+1] − RR[n], length N-1

    # Brennan (2001) Eq. 4: SD1² = 0.5 × Var(diff_rr), ddof=1
    sd1 = float(np.sqrt(0.5 * np.var(diff_rr, ddof=1)))

    # Brennan (2001) Eq. 5: SD2² = 2 × SDNN² − SD1²
    sdnn = float(np.std(rr_ms, ddof=1))
    sd2_sq = 2.0 * sdnn**2 - sd1**2
    sd2 = float(np.sqrt(max(sd2_sq, 0.0)))   # guard against float rounding

    sd1_sd2 = sd1 / sd2 if sd2 > 0 else float("nan")
    area    = float(np.pi * sd1 * sd2)

    return {
        "sd1_ms":   sd1,
        "sd2_ms":   sd2,
        "sd1_sd2":  sd1_sd2,
        "area_ms2": area,
        "rr_n":     rr_ms[:-1],
        "rr_n1":    rr_ms[1:],
    }
