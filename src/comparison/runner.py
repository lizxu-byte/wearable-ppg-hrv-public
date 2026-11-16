"""
runner.py — End-to-end PPG algorithm comparison on BIDMC dataset (P3-2)

Runs all combinations of peak-detection × RR-cleaning methods on BIDMC records
and computes three metrics against ECG ground truth:

    HR MAE    Mean absolute error of 30-second window HR estimates (bpm).
              Captures missed and spurious beats; robust to pulse arrival time.

    RMSSD Δ   Absolute error of full-session RMSSD vs ECG reference (ms).
              Captures IBI timing accuracy. HR MAE can be good while RMSSD
              is terrible (HR MAE is the "known blind spot" for this project).

    F1        Beat-level precision/recall with PAT-aware matching.
              PAT (pulse arrival time) is estimated per record as the median
              lead time from ECG R-peak to the nearest subsequent PPG peak.
              Matching tolerance: ±150 ms around expected PPG peak position.

Fixed preprocessing (from P1-3 and P1-4 validation):
    Bandpass: Butterworth 0.4–10 Hz, order 4, zero-phase (sosfiltfilt).
    Selected: lowest peak-timing MAE and good stopband attenuation.

Method grid:
    Peak detection  :  elgendi (A), prominence (C)
                       (bishop (B) excluded on BIDMC — see PEAK_METHODS below)
    RR cleaning     :  physiological_3sd (A), malik (B), karlsson (C)
    → 2 × 3 = 6 combinations

References:
    Elgendi M et al. (2013). PLoS ONE 8(10):e76585.
    Bishop SM, Ercole A (2018). Front Neuroinform 12:62.
    Malik M et al. (1996). Eur Heart J 17:354–381.  (Malik RR rule)
    Karlsson M et al. (2012). Biomed Eng Online 11:2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pipeline.filter import bandpass_butterworth
from src.pipeline.peaks import detect_peaks
from src.pipeline.rr_clean import clean_rr
from src.validation.public_dataset import BIDMCRecord


# ── Constants ─────────────────────────────────────────────────────────────────

FILTER_LOWCUT  = 0.4   # Hz  (validated P1-4: recommended 0.4–10 Hz order 4)
FILTER_HIGHCUT = 10.0  # Hz
FILTER_ORDER   = 4

WINDOW_S        = 30.0   # seconds per HR window
PAT_TOLERANCE_S = 0.150  # ±150 ms matching tolerance after PAT correction
PAT_MIN_S       = 0.050  # minimum plausible PAT
PAT_MAX_S       = 0.800  # maximum plausible PAT

# Bishop (neurokit2) is excluded from BIDMC comparison: its O(N²) complexity
# (scales from 1 to N/2, scanning full signal at each scale) makes it
# impractical for 8-minute recordings at 125 Hz (N=60,000 → ~10 h/record).
# It was validated on 30-second synthetic segments in test_peaks.py (F1=0.993).
# For long recordings use Elgendi (O(N)) or Prominence (O(N log N)).
PEAK_METHODS  = ["elgendi", "prominence"]
CLEAN_METHODS = ["physiological_3sd", "malik", "karlsson"]

ALL_COMBINATIONS = [
    (p, c) for p in PEAK_METHODS for c in CLEAN_METHODS
]


# ── Per-record result ─────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    record_id:      str
    peak_method:    str
    clean_method:   str
    hr_mae_bpm:     float   # mean |HR_ppg - HR_ecg| over 30s windows (bpm)
    hr_rmse_bpm:    float   # RMS version
    rmssd_error_ms: float   # |RMSSD_ppg - RMSSD_ecg| (ms)
    f1:             float   # beat F1 with PAT correction
    precision:      float
    recall:         float
    pat_s:          float   # estimated pulse arrival time (s)
    n_ppg_peaks:    int
    n_ecg_peaks:    int
    n_clean_rr:     int     # usable RR after cleaning
    failed:         bool = False
    error_msg:      str  = ""


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_pipeline(
    record: BIDMCRecord,
    peak_method: str = "elgendi",
    clean_method: str = "karlsson",
) -> PipelineResult:
    """
    Run one pipeline combination on a single BIDMC record.

    Parameters
    ----------
    record       : BIDMCRecord  Loaded BIDMC record with ECG ground truth.
    peak_method  : str  "elgendi", "bishop", or "prominence".
    clean_method : str  "physiological_3sd", "malik", or "karlsson".

    Returns
    -------
    PipelineResult  All computed metrics, or failed=True on error.
    """
    def _fail(msg: str) -> PipelineResult:
        return PipelineResult(
            record_id=record.record_id, peak_method=peak_method,
            clean_method=clean_method,
            hr_mae_bpm=float("nan"), hr_rmse_bpm=float("nan"),
            rmssd_error_ms=float("nan"), f1=float("nan"),
            precision=float("nan"), recall=float("nan"),
            pat_s=float("nan"), n_ppg_peaks=0, n_ecg_peaks=len(record.r_peaks),
            n_clean_rr=0, failed=True, error_msg=msg,
        )

    try:
        fs = record.fs

        # ── 1. Bandpass filter PPG ────────────────────────────────────────────
        ppg_filt = bandpass_butterworth(
            record.ppg, fs,
            lowcut_hz=FILTER_LOWCUT, highcut_hz=FILTER_HIGHCUT, order=FILTER_ORDER,
        )

        # ── 2. Detect PPG peaks ───────────────────────────────────────────────
        ppg_peaks = detect_peaks(ppg_filt, fs, method=peak_method)
        if len(ppg_peaks) < 5:
            return _fail(f"too few PPG peaks: {len(ppg_peaks)}")

        # ── 3. PPG RR intervals → clean ───────────────────────────────────────
        ppg_rr_raw = np.diff(ppg_peaks) / fs * 1000.0   # ms
        ppg_rr, _  = clean_rr(ppg_rr_raw, method=clean_method)
        if len(ppg_rr) < 5:
            return _fail(f"too few RR after cleaning: {len(ppg_rr)}")

        # ── 4. HR MAE per 30-second window ───────────────────────────────────
        hr_mae_bpm, hr_rmse_bpm = _hr_per_window(
            ppg_peaks, record.r_peaks, fs, WINDOW_S
        )

        # ── 5. RMSSD error ───────────────────────────────────────────────────
        ppg_rmssd   = float(np.sqrt(np.mean(np.diff(ppg_rr) ** 2)))
        ecg_rmssd   = record.rmssd_ms
        rmssd_error = abs(ppg_rmssd - ecg_rmssd) if np.isfinite(ecg_rmssd) else float("nan")

        # ── 6. Beat-level F1 with PAT correction ─────────────────────────────
        f1, precision, recall, pat_s = _beat_f1_with_pat(
            ppg_peaks, record.r_peaks, fs, PAT_TOLERANCE_S
        )

        return PipelineResult(
            record_id=record.record_id,
            peak_method=peak_method, clean_method=clean_method,
            hr_mae_bpm=hr_mae_bpm, hr_rmse_bpm=hr_rmse_bpm,
            rmssd_error_ms=rmssd_error,
            f1=f1, precision=precision, recall=recall,
            pat_s=pat_s,
            n_ppg_peaks=len(ppg_peaks), n_ecg_peaks=len(record.r_peaks),
            n_clean_rr=len(ppg_rr),
        )

    except Exception as exc:
        return _fail(str(exc)[:80])


# ── Metric helpers ────────────────────────────────────────────────────────────

def _hr_per_window(
    ppg_peaks: np.ndarray,
    ecg_peaks: np.ndarray,
    fs: int,
    window_s: float,
) -> tuple[float, float]:
    """
    Compute per-window HR MAE and RMSE between PPG and ECG.

    HR for each window is estimated as: 60 / mean(RR_ms) × 1000.
    Windows with fewer than 3 beats in either PPG or ECG are skipped.
    """
    n_samples   = max(ppg_peaks[-1], ecg_peaks[-1]) + 1
    win_samples = int(window_s * fs)
    n_windows   = n_samples // win_samples

    hr_errors = []
    for i in range(n_windows):
        t0, t1 = i * win_samples, (i + 1) * win_samples
        ppg_w = ppg_peaks[(ppg_peaks >= t0) & (ppg_peaks < t1)]
        ecg_w = ecg_peaks[(ecg_peaks >= t0) & (ecg_peaks < t1)]
        if len(ppg_w) < 3 or len(ecg_w) < 3:
            continue
        hr_ppg = 60_000.0 / float(np.mean(np.diff(ppg_w) / fs * 1000.0))
        hr_ecg = 60_000.0 / float(np.mean(np.diff(ecg_w) / fs * 1000.0))
        hr_errors.append(abs(hr_ppg - hr_ecg))

    if not hr_errors:
        return float("nan"), float("nan")
    return float(np.mean(hr_errors)), float(np.sqrt(np.mean(np.array(hr_errors) ** 2)))


def _beat_f1_with_pat(
    ppg_peaks: np.ndarray,
    ecg_peaks: np.ndarray,
    fs: int,
    tolerance_s: float,
) -> tuple[float, float, float, float]:
    """
    Compute beat-level F1 after estimating and correcting for PAT.

    PAT estimation: for each ECG R-peak find the first subsequent PPG peak
    in [PAT_MIN_S, PAT_MAX_S] using binary search. O(N log N).

    Matching: for each ECG peak compute expected PPG time = ecg_t + PAT,
    then find the nearest PPG peak using np.searchsorted. Check only the
    two nearest candidates (left and right). O(N log N) total.

    Returns (f1, precision, recall, pat_s).
    """
    ppg_t = ppg_peaks / fs   # seconds, already sorted
    ecg_t = ecg_peaks / fs

    # ── PAT estimation — O(N log N) ─────────────────────────────────────────
    delays = []
    for et in ecg_t:
        lo = int(np.searchsorted(ppg_t, et + PAT_MIN_S))
        if lo < len(ppg_t) and ppg_t[lo] <= et + PAT_MAX_S:
            delays.append(ppg_t[lo] - et)
    pat_s = float(np.median(delays)) if delays else 0.25

    # ── Greedy matching — O(N log N) ────────────────────────────────────────
    # For each ECG peak, expected PPG position = ecg_t + pat_s.
    # Use searchsorted to find the two nearest PPG peaks and pick the closer.
    used_ppg = np.zeros(len(ppg_t), dtype=bool)
    tp = 0

    expected_ppg = ecg_t + pat_s
    for exp in expected_ppg:
        idx = int(np.searchsorted(ppg_t, exp))
        best_idx  = -1
        best_dist = tolerance_s + 1.0
        for i in (idx - 1, idx):
            if 0 <= i < len(ppg_t) and not used_ppg[i]:
                d = abs(ppg_t[i] - exp)
                if d < best_dist:
                    best_dist = d
                    best_idx  = i
        if best_idx >= 0 and best_dist <= tolerance_s:
            used_ppg[best_idx] = True
            tp += 1

    fp = len(ppg_peaks) - tp
    fn = len(ecg_peaks) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return f1, precision, recall, pat_s


# ── Batch comparison ──────────────────────────────────────────────────────────

def run_comparison(
    records: list[BIDMCRecord],
    combinations: list[tuple[str, str]] | None = None,
) -> list[PipelineResult]:
    """
    Run all method combinations on all records.

    Parameters
    ----------
    records      : list[BIDMCRecord]  Loaded BIDMC records.
    combinations : list of (peak_method, clean_method) pairs.
                   Defaults to ALL_COMBINATIONS (6 pairs).

    Returns
    -------
    list[PipelineResult]  One entry per (record × combination).
    """
    if combinations is None:
        combinations = ALL_COMBINATIONS

    results = []
    for rec in records:
        for peak_m, clean_m in combinations:
            res = evaluate_pipeline(rec, peak_m, clean_m)
            results.append(res)
    return results


def aggregate_results(results: list[PipelineResult]) -> dict[tuple[str, str], dict]:
    """
    Aggregate per-record results per (peak_method, clean_method) combination.

    Returns a dict keyed by (peak_method, clean_method) with aggregated
    metrics: median and IQR of HR MAE, RMSSD error, and F1 across records.
    Skips failed records.
    """
    from collections import defaultdict

    buckets: dict[tuple, list] = defaultdict(list)
    for r in results:
        if not r.failed:
            buckets[(r.peak_method, r.clean_method)].append(r)

    agg = {}
    for combo, recs in buckets.items():
        hr_maes   = [r.hr_mae_bpm     for r in recs if np.isfinite(r.hr_mae_bpm)]
        rmssd_errs = [r.rmssd_error_ms for r in recs if np.isfinite(r.rmssd_error_ms)]
        f1s       = [r.f1             for r in recs if np.isfinite(r.f1)]

        agg[combo] = {
            "n_records":        len(recs),
            "n_failed":         sum(1 for r in results
                                    if (r.peak_method, r.clean_method) == combo and r.failed),
            "hr_mae_median":    float(np.median(hr_maes))    if hr_maes   else float("nan"),
            "hr_mae_p75":       float(np.percentile(hr_maes, 75)) if hr_maes else float("nan"),
            "rmssd_err_median": float(np.median(rmssd_errs)) if rmssd_errs else float("nan"),
            "rmssd_err_p75":    float(np.percentile(rmssd_errs, 75)) if rmssd_errs else float("nan"),
            "f1_median":        float(np.median(f1s))        if f1s       else float("nan"),
            "f1_p25":           float(np.percentile(f1s, 25)) if f1s      else float("nan"),
        }
    return agg


def best_combination(agg: dict[tuple[str, str], dict]) -> tuple[str, str]:
    """
    Select the best (peak_method, clean_method) combination.

    Ranking criterion: minimize HR MAE median (primary), then RMSSD error
    median (secondary). Both metrics must be finite.
    """
    valid = {k: v for k, v in agg.items()
             if np.isfinite(v["hr_mae_median"]) and np.isfinite(v["rmssd_err_median"])}
    if not valid:
        raise ValueError("No valid combinations to rank")
    return min(valid, key=lambda k: (valid[k]["hr_mae_median"], valid[k]["rmssd_err_median"]))
