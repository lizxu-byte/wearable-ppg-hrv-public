"""
rr_clean.py — RR interval artifact cleaning methods

RR cleaning is a bilateral error source:
    False removal  → artificially short/missing sequences → RMSSD and SDNN too LOW
    Missed artifact → abnormal RR intervals stay → RMSSD and SDNN too HIGH
Both destroy HRV validity; the cleaning method must balance these two error types.

Three methods implemented:
    A. physiological_3sd — hard physiological bounds + iterative 3-SD removal
    B. malik             — adjacent-beat relative change threshold (Malik 1996)
    C. karlsson          — iterative local median deviation threshold

All methods return a boolean mask and cleaned RR array (not interpolated).
Interpolation of removed intervals is handled separately in rr_interpolate.py.

References:
    A. Task Force of the European Society of Cardiology (1996). Heart rate
       variability: standards of measurement, physiological interpretation
       and clinical use. Circulation 93:1043-1065.
       https://doi.org/10.1161/01.CIR.93.5.1043
       (Physiological RR range and 3SD criterion)

    B. Malik M et al. (1996). Heart rate variability: Standards of
       measurement, physiological interpretation, and clinical use.
       Eur Heart J 17:354-381.
       (Adjacent-beat relative change threshold; 20% widely cited)

    C. Karlsson M et al. (2012). Automatic filtering of outliers in RR
       intervals before analysis of heart rate variability in Holter
       recordings: a comparison with carefully edited data.
       Biomed Eng Online 11:2.
       https://doi.org/10.1186/1475-925X-11-2
       (Iterative median-based filter; Table 1: threshold=0.20, window=5)
"""

from __future__ import annotations

import numpy as np


# ── Method A: Physiological bounds + iterative 3-SD ──────────────────────────

def clean_physiological_3sd(
    rr_ms: np.ndarray,
    rr_min_ms: float = 400.0,
    rr_max_ms: float = 1500.0,
    n_sd: float = 3.0,
    max_iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove RR intervals outside physiological bounds, then iteratively remove
    intervals beyond n_sd standard deviations from the remaining mean.

    Two-pass approach:
        Pass 1 — hard bounds: removes extreme values that would inflate SD.
        Pass 2 — iterative 3SD: converges to stable mean/SD without outliers.

    Reference: Task Force ESC (1996) Circulation 93:1043-1065.
               Physiological range 400–1500 ms corresponds to HR 40–150 bpm.

    Parameters
    ----------
    rr_ms       : np.ndarray  Raw RR intervals (ms), 1-D.
    rr_min_ms   : float       Lower physiological bound (ms). Default 400 ms (150 bpm).
                              For athletes: 333 ms (180 bpm).
    rr_max_ms   : float       Upper physiological bound (ms). Default 1500 ms (40 bpm).
                              For bradycardia studies: 2000 ms.
    n_sd        : float       SD multiplier for iterative removal. Default 3.0.
    max_iterations: int       Safety cap on iterations. Default 20.

    Returns
    -------
    rr_clean : np.ndarray  Cleaned RR intervals (ms). May be shorter than input.
    mask     : np.ndarray  Boolean mask, True = kept. len == len(rr_ms).
    """
    rr_ms = np.asarray(rr_ms, dtype=float)
    mask = np.ones(len(rr_ms), dtype=bool)

    # Pass 1: physiological bounds
    mask &= (rr_ms >= rr_min_ms) & (rr_ms <= rr_max_ms)

    # Pass 2: iterative SD
    for _ in range(max_iterations):
        current = rr_ms[mask]
        if len(current) < 3:
            break
        m, s = np.mean(current), np.std(current)
        new_mask = mask.copy()
        new_mask[mask] &= np.abs(current - m) <= n_sd * s
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    return rr_ms[mask], mask


# ── Method B: Malik rule ──────────────────────────────────────────────────────

def clean_malik(
    rr_ms: np.ndarray,
    rr_min_ms: float = 400.0,
    rr_max_ms: float = 1500.0,
    threshold: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Malik rule: flag RR[n] if it differs from RR[n-1] by more than threshold
    fraction, OR if it lies outside physiological bounds.

    The Malik rule is a local criterion — it flags abrupt changes regardless
    of the overall distribution, making it more sensitive to single ectopic
    beats (isolated PVC/PAC) than the global 3-SD method.

    Reference: Malik M et al. (1996) Eur Heart J 17:354-381.
               Task Force ESC (1996) Circulation 93:1043-1065.

    Parameters
    ----------
    rr_ms     : np.ndarray  Raw RR intervals (ms), 1-D.
    rr_min_ms : float       Lower physiological bound (ms).
    rr_max_ms : float       Upper physiological bound (ms).
    threshold : float       Relative change threshold. Default 0.20 (20%).
                            For noisier signals (motion): raise to 0.25.
                            For Holter ECG reference: 0.20 (Malik 1996 standard).

    Returns
    -------
    rr_clean : np.ndarray  Cleaned RR intervals.
    mask     : np.ndarray  Boolean mask, True = kept.
    """
    rr_ms = np.asarray(rr_ms, dtype=float)
    mask = np.ones(len(rr_ms), dtype=bool)

    # Physiological bounds
    mask &= (rr_ms >= rr_min_ms) & (rr_ms <= rr_max_ms)

    # Malik adjacent-beat criterion
    # Flag RR[n] if |RR[n] - RR[n-1]| / RR[n-1] > threshold
    # Use the original (unmasked) sequence for adjacent comparison to preserve index alignment
    rel_diff = np.abs(np.diff(rr_ms)) / (rr_ms[:-1] + 1e-9)
    # Flag both the changed beat and the preceding beat? No — standard is to flag only RR[n].
    flagged = np.concatenate([[False], rel_diff > threshold])
    mask &= ~flagged

    return rr_ms[mask], mask


# ── Method C: Karlsson 2012 iterative median ─────────────────────────────────

def clean_karlsson(
    rr_ms: np.ndarray,
    rr_min_ms: float = 400.0,
    rr_max_ms: float = 1500.0,
    threshold: float = 0.20,
    window: int = 5,
    max_iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Karlsson 2012 iterative median-based outlier removal.

    For each RR[n], compute the local median of the surrounding window
    (window beats). Flag RR[n] if |RR[n] - median_local| / median_local > threshold.
    Iterate on the remaining intervals until convergence.

    Compared to 3SD: robust to non-Gaussian distributions (common in real data).
    Compared to Malik: evaluates each beat against a local context, not just
    its predecessor — more robust to gradual HR changes.

    Reference: Karlsson M et al. (2012) Biomed Eng Online 11:2.
               Table 1: threshold=0.20, window=5 beats yielded best performance
               in their Holter ECG benchmark.

    Parameters
    ----------
    rr_ms       : np.ndarray  Raw RR intervals (ms), 1-D.
    rr_min_ms   : float       Lower physiological bound (ms).
    rr_max_ms   : float       Upper physiological bound (ms).
    threshold   : float       Relative deviation from local median. Default 0.20.
                              Karlsson (2012) Table 1: 0.20 is optimal across datasets.
                              For high-activity subjects: raise to 0.25.
    window      : int         Local median window (beats). Default 5.
                              Karlsson (2012) Table 1: 5 beats optimal.

    Returns
    -------
    rr_clean : np.ndarray  Cleaned RR intervals.
    mask     : np.ndarray  Boolean mask, True = kept.
    """
    rr_ms = np.asarray(rr_ms, dtype=float)
    mask = np.ones(len(rr_ms), dtype=bool)

    # Physiological bounds first
    mask &= (rr_ms >= rr_min_ms) & (rr_ms <= rr_max_ms)

    half_w = window // 2

    for _ in range(max_iterations):
        indices = np.where(mask)[0]
        if len(indices) < window:
            break
        new_mask = mask.copy()
        current = rr_ms[indices]

        for k, idx in enumerate(indices):
            lo = max(0, k - half_w)
            hi = min(len(current), k + half_w + 1)
            # Exclude self from median computation
            context = np.concatenate([current[lo:k], current[k + 1:hi]])
            if len(context) == 0:
                continue
            median_local = np.median(context)
            if median_local < 1.0:
                continue
            if abs(current[k] - median_local) / median_local > threshold:
                new_mask[idx] = False

        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    return rr_ms[mask], mask


# ── Dispatcher ────────────────────────────────────────────────────────────────

_METHODS = {
    "physiological_3sd": clean_physiological_3sd,
    "malik":             clean_malik,
    "karlsson":          clean_karlsson,
}


def clean_rr(
    rr_ms: np.ndarray,
    method: str,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch to the selected RR cleaning method.

    Parameters
    ----------
    rr_ms  : np.ndarray  Raw RR intervals (ms).
    method : str         One of "physiological_3sd", "malik", "karlsson".
    **kwargs             Forwarded to the method function.

    Returns
    -------
    rr_clean : np.ndarray  Cleaned RR intervals (ms).
    mask     : np.ndarray  Boolean mask (True = kept).
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown RR cleaning method '{method}'. Choose from {list(_METHODS)}")
    return _METHODS[method](rr_ms, **kwargs)


# ── Evaluation metrics ────────────────────────────────────────────────────────

def evaluate(
    rr_clean: np.ndarray,
    rr_true: np.ndarray,
    mask: np.ndarray,
    ectopic_mask: np.ndarray,
) -> dict:
    """
    Evaluate RR cleaning quality given known ground truth and injected ectopic positions.

    Parameters
    ----------
    rr_clean     : np.ndarray  Cleaned RR array returned by clean_* function.
    rr_true      : np.ndarray  Ground-truth clean RR array (no ectopics).
    mask         : np.ndarray  Boolean mask from clean_*: True = kept.
    ectopic_mask : np.ndarray  Boolean mask: True = this RR is an injected ectopic.

    Returns
    -------
    dict with keys:
        n_total          — total input RR count
        n_kept           — count kept after cleaning
        deletion_rate    — fraction removed (lower = more conservative)
        ectopic_recall   — TP_ectopics / (TP_ectopics + FN_ectopics) → detection power
        normal_ppv       — TN_normals / (TN_normals + FP_normals)    → preservation
        rmssd_true_ms    — RMSSD of ground-truth clean RR
        rmssd_clean_ms   — RMSSD of cleaned RR
        rmssd_error_ms   — |rmssd_clean - rmssd_true|
        sdnn_true_ms     — SDNN of ground-truth clean RR
        sdnn_clean_ms    — SDNN of cleaned RR
        sdnn_error_ms    — |sdnn_clean - sdnn_true|
        meannn_true_ms   — mean RR of ground truth
        meannn_clean_ms  — mean RR after cleaning
        meannn_error_ms  — |meannn_clean - meannn_true|
    """
    mask = np.asarray(mask, dtype=bool)
    ectopic_mask = np.asarray(ectopic_mask, dtype=bool)
    n_total = len(mask)

    removed_mask = ~mask   # True = removed

    # Ectopic detection
    tp = int(np.sum(removed_mask & ectopic_mask))    # correctly removed ectopics
    fn = int(np.sum(~removed_mask & ectopic_mask))   # missed ectopics (kept)
    fp = int(np.sum(removed_mask & ~ectopic_mask))   # normal beats incorrectly removed

    ectopic_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    normal_ppv     = (n_total - int(np.sum(ectopic_mask)) - fp) / max(1, n_total - int(np.sum(ectopic_mask)))

    def hrv_metrics(rr):
        if len(rr) < 2:
            return float("nan"), float("nan"), float("nan")
        rr_diff = np.diff(rr.astype(float))
        rmssd = float(np.sqrt(np.mean(rr_diff ** 2)))
        sdnn  = float(np.std(rr.astype(float), ddof=1))
        meannn = float(np.mean(rr))
        return rmssd, sdnn, meannn

    rmssd_true,  sdnn_true,  mean_true  = hrv_metrics(rr_true)
    rmssd_clean, sdnn_clean, mean_clean = hrv_metrics(rr_clean)

    return {
        "n_total":         n_total,
        "n_kept":          int(np.sum(mask)),
        "deletion_rate":   float(np.sum(removed_mask)) / n_total,
        "ectopic_recall":  ectopic_recall,
        "normal_ppv":      normal_ppv,
        "tp":              tp,
        "fp":              fp,
        "fn":              fn,
        "rmssd_true_ms":   rmssd_true,
        "rmssd_clean_ms":  rmssd_clean,
        "rmssd_error_ms":  abs(rmssd_clean - rmssd_true) if np.isfinite(rmssd_clean) else float("nan"),
        "sdnn_true_ms":    sdnn_true,
        "sdnn_clean_ms":   sdnn_clean,
        "sdnn_error_ms":   abs(sdnn_clean - sdnn_true) if np.isfinite(sdnn_clean) else float("nan"),
        "meannn_true_ms":  mean_true,
        "meannn_clean_ms": mean_clean,
        "meannn_error_ms": abs(mean_clean - mean_true) if np.isfinite(mean_clean) else float("nan"),
    }


# ── Comparison runner ─────────────────────────────────────────────────────────

def compare_methods(
    rr_contaminated: np.ndarray,
    rr_true: np.ndarray,
    ectopic_mask: np.ndarray,
    configs: list[dict],
) -> list[dict]:
    """
    Run multiple RR cleaning methods and evaluate each against ground truth.

    Parameters
    ----------
    rr_contaminated : np.ndarray  RR sequence with injected ectopics.
    rr_true         : np.ndarray  Ground-truth clean RR sequence.
    ectopic_mask    : np.ndarray  Boolean, True = this beat is an injected ectopic.
    configs         : list[dict]  Each: {name, method, **kwargs}.

    Returns
    -------
    list[dict]  Evaluation metrics per method.
    """
    results = []
    for cfg in configs:
        name   = cfg["name"]
        method = cfg["method"]
        kwargs = {k: v for k, v in cfg.items() if k not in ("name", "method")}
        row = {"name": name, "method": method}
        try:
            rr_clean, mask = clean_rr(rr_contaminated, method, **kwargs)
            row.update(evaluate(rr_clean, rr_true, mask, ectopic_mask))
        except Exception as e:
            row["error"] = str(e)
        results.append(row)
    return results
