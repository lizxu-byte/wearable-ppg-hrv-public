"""
gsr.py — Electrodermal Activity (EDA / GSR) analysis

Decomposes the raw skin conductance signal (µS) into two components:

    SCL — Skin Conductance Level (tonic / slow)
          Reflects baseline sympathetic nervous system activity.
          Extracted as the low-frequency envelope (< 0.05 Hz).

    SCR — Skin Conductance Response (phasic / fast)
          Reflects discrete arousal events (task, stimulus, stress).
          Extracted as phasic = total − SCL; peaks detected above threshold.

Signal model (Boucsein 2012):
    EDA(t) = SCL(t) + SCR(t)
    SCL(t) = LP₀.₀₅(EDA)           — Butterworth, 0.05 Hz, zero-phase
    SCR(t) = EDA(t) − SCL(t)        — residual phasic component

Error propagation (Algorithm Engineering Discipline Rule 2):
    Lowpass cutoff too high → SCR bleeds into SCL → amplitude underestimated.
    Lowpass cutoff too low → slow SCR tails remain in phasic → double-counting.
    0.05 Hz is the Boucsein 2012 convention; appropriate for typical 2–15 s SCRs.

References:
    Boucsein W et al. (2012). Publication recommendations for electrodermal
    measurements. Psychophysiology 49(8):1017-1034.
    https://doi.org/10.1111/j.1469-8986.2012.01384.x
    (Gold standard reference; defines SCL/SCR decomposition, minimum amplitude
    0.01 µS, rise-time criteria 1–5 s)

    Braithwaite JJ et al. (2013). A guide for analysing electrodermal activity
    (EDA) & skin conductance responses (SCRs) for psychological experiments.
    Psychophysiology 49(8):1017-1034.
    (Practical implementation guide for SCR detection criteria)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks


# ── SCL extraction ────────────────────────────────────────────────────────────

def extract_scl(
    gsr_us: np.ndarray,
    fs: float,
    lowpass_hz: float = 0.05,
    order: int = 4,
) -> np.ndarray:
    """
    Extract Skin Conductance Level (tonic component) via zero-phase Butterworth.

    SCL represents slow baseline sympathetic activity. Values below 0.05 Hz
    are considered tonic (Boucsein 2012, Section 2.1).

    Parameters
    ----------
    gsr_us    : np.ndarray  Raw EDA signal (µS). 1-D, ≥ 2 samples.
    fs        : float       Sampling rate (Hz). Typically 256.
    lowpass_hz: float       Low-pass cutoff (Hz). Default 0.05.
                            Boucsein (2012): tonic/phasic boundary at 0.05 Hz.
                            For slowly changing skin temperature artifacts:
                            lower to 0.02 Hz. Raising above 0.1 Hz risks
                            contaminating SCL with SCR tails.
    order     : int         Butterworth order. Default 4.

    Returns
    -------
    np.ndarray  SCL time series (µS), same length as gsr_us.
    """
    gsr = np.asarray(gsr_us, dtype=float)
    sos = butter(order, lowpass_hz, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, gsr)


# ── SCR extraction (phasic component) ────────────────────────────────────────

def extract_scr(
    gsr_us: np.ndarray,
    scl: np.ndarray,
) -> np.ndarray:
    """
    Extract phasic Skin Conductance Response: SCR = EDA − SCL.

    The phasic component contains the rapid deflections corresponding to
    discrete arousal events. It oscillates around zero with positive
    deflections marking SCR events.

    Parameters
    ----------
    gsr_us : np.ndarray  Raw EDA signal (µS).
    scl    : np.ndarray  SCL from extract_scl(), same length.

    Returns
    -------
    np.ndarray  Phasic SCR signal (µS), same length.
    """
    return np.asarray(gsr_us, dtype=float) - np.asarray(scl, dtype=float)


# ── SCR event detection ───────────────────────────────────────────────────────

def detect_scr_events(
    scr_signal: np.ndarray,
    fs: float,
    min_amplitude_us: float = 0.01,
    min_rise_s: float = 0.5,
    max_rise_s: float = 10.0,
    min_interval_s: float = 1.0,
) -> list[dict]:
    """
    Detect discrete SCR events in the phasic signal.

    Algorithm:
        1. Find local maxima (peaks) using prominence ≥ min_amplitude_us.
        2. For each peak, search backward up to max_rise_s to find the
           preceding local minimum (SCR onset).
        3. Amplitude = signal[peak] − signal[onset].
        4. Keep only events with amplitude ≥ min_amplitude_us and
           rise time in [min_rise_s, max_rise_s].

    Reference: Boucsein et al. (2012) Psychophysiology 49(8):1017-1034.
               Minimum amplitude 0.01 µS; rise time 1–5 s typical.

    Parameters
    ----------
    scr_signal      : np.ndarray  Phasic SCR signal (µS) from extract_scr().
    fs              : float       Sampling rate (Hz).
    min_amplitude_us: float       Minimum SCR amplitude (µS). Default 0.01.
                                  Boucsein (2012): 0.01 µS is the minimum
                                  resolvable response. Raise to 0.05 µS for
                                  high-noise recordings to reduce false positives.
    min_rise_s      : float       Minimum onset-to-peak rise time (s). Default 0.5.
                                  Excludes noise spikes (< 0.5 s rise).
    max_rise_s      : float       Maximum onset-to-peak rise time (s). Default 10.
                                  Boucsein (2012): typical spontaneous SCR rise ≤ 5 s;
                                  10 s allows for slow responses in resting conditions.
    min_interval_s  : float       Minimum time between successive peaks (s). Default 1.

    Returns
    -------
    list of dict, each with keys:
        onset_idx       : int    sample index of SCR onset
        peak_idx        : int    sample index of SCR peak
        onset_s         : float  onset time (seconds from start)
        peak_s          : float  peak time (seconds)
        amplitude_us    : float  SCR amplitude (µS)
        rise_time_s     : float  onset → peak duration (s)
    """
    scr = np.asarray(scr_signal, dtype=float)
    min_dist         = max(1, int(min_interval_s * fs))
    min_rise_samples = max(1, int(min_rise_s * fs))

    # Smooth phasic to suppress noise-induced micro-peaks before detection.
    # A 0.25 s moving average removes oscillations above ~4 Hz while preserving
    # the 0.2–1 Hz SCR shape (Braithwaite 2013: smooth EDA before detecting SCRs).
    # Peak positions and onset locations are found on the smoothed signal;
    # amplitudes are measured on the original signal to avoid bias.
    smooth_n   = max(3, int(0.25 * fs))
    kernel     = np.ones(smooth_n) / smooth_n
    scr_smooth = np.convolve(scr, kernel, mode="same")

    peak_idxs, _ = find_peaks(
        scr_smooth,
        prominence=min_amplitude_us / 2.0,   # permissive; amplitude gate below
        distance=min_dist,
    )

    # Pre-event baseline constants: mean of original phasic in
    # [peak − BASELINE_PRE_S, peak − BASELINE_POST_S].
    # Amplitude = smoothed_peak − pre_event_baseline is more robust than
    # peak − onset for two reasons:
    #   1. sosfiltfilt zero-phase LP "anticipates" upcoming SCRs, raising the
    #      SCL before the event onset, which depresses the phasic monotonically
    #      in the search window.  argmin then lands mid-rise instead of at the
    #      true pre-event baseline, giving a near-zero amplitude (Benedek &
    #      Kaernbach 2010, Fig. 4).
    #   2. Amplitude referenced to a stable window is invariant to onset
    #      estimation errors in spontaneous SCRs with no clear stimulus marker.
    # Reference: Benedek M & Kaernbach C (2010). A continuous measure of phasic
    # electrodermal activity. J Neurosci Methods 190(1):80-91.
    # https://doi.org/10.1016/j.jneumeth.2010.04.028
    BASELINE_PRE_S  = 15.0
    BASELINE_POST_S =  5.0

    events: list[dict] = []
    for peak_idx in peak_idxs:
        # Skip events that are too close to the start of the signal to have a
        # valid baseline window.  A peak at t < BASELINE_PRE_S is unreliable:
        # the LP-filter has not settled (sosfiltfilt edge artifact) and the
        # baseline window would cover only a fraction of BASELINE_PRE_S.
        if peak_idx < int(BASELINE_PRE_S * fs):
            continue

        # ── Amplitude (baseline-corrected) ───────────────────────────────────
        bl_start     = max(0, peak_idx - int(BASELINE_PRE_S  * fs))
        bl_end       = max(bl_start + 1, peak_idx - int(BASELINE_POST_S * fs))
        baseline_val = float(np.mean(scr[bl_start:bl_end]))
        amplitude    = float(scr_smooth[peak_idx] - baseline_val)

        if amplitude < min_amplitude_us:
            continue

        # ── Onset detection (rise-time only) ─────────────────────────────────
        search_start  = max(0, peak_idx - int(max_rise_s * fs))
        window_smooth = scr_smooth[search_start : peak_idx + 1]

        if len(window_smooth) < 3:
            continue

        # Find the most recent local minimum in the smoothed window at least
        # min_rise_s before the peak.  Fallback: use onset_rel=0 (earliest
        # possible onset = peak − max_rise_s) to avoid the argmin landing
        # mid-rise due to LP anticipation artifact (see note above).
        diffs = np.diff(window_smooth)
        if len(diffs) >= 2:
            neg_to_pos    = (diffs[:-1] <= 0) & (diffs[1:] > 0)
            lm_idxs       = np.where(neg_to_pos)[0] + 1
            max_onset_rel = len(window_smooth) - 1 - min_rise_samples
            valid_lm      = lm_idxs[lm_idxs <= max_onset_rel]
            onset_rel     = int(valid_lm[-1]) if len(valid_lm) > 0 else 0
        else:
            onset_rel = 0

        onset_idx = search_start + onset_rel
        rise_time = (peak_idx - onset_idx) / fs

        if not (min_rise_s <= rise_time <= max_rise_s):
            continue

        events.append({
            "onset_idx":    onset_idx,
            "peak_idx":     peak_idx,
            "onset_s":      onset_idx / fs,
            "peak_s":       peak_idx / fs,
            "amplitude_us": amplitude,
            "rise_time_s":  rise_time,
        })

    return events


# ── Full feature extraction ───────────────────────────────────────────────────

def compute_gsr_features(
    gsr_us: np.ndarray,
    fs: float,
    lowpass_hz: float = 0.05,
    min_amplitude_us: float = 0.01,
    min_rise_s: float = 0.5,
    max_rise_s: float = 10.0,
    min_interval_s: float = 1.0,
) -> dict:
    """
    Compute all GSR / EDA features from a single recording segment.

    Parameters
    ----------
    gsr_us          : np.ndarray  Raw EDA signal (µS). Should be ≥ 60 s for
                                  reliable SCL estimation (needs > 3 / lowpass_hz
                                  seconds for the filter to settle).
    fs              : float       Sampling rate (Hz).
    lowpass_hz      : float       SCL / SCR boundary (Hz). Default 0.05.
    min_amplitude_us: float       Minimum SCR amplitude (µS). Default 0.01.
    min_rise_s      : float       Minimum SCR rise time (s). Default 0.5.
    max_rise_s      : float       Maximum SCR rise time (s). Default 10.
    min_interval_s  : float       Minimum inter-SCR interval (s). Default 1.

    Returns
    -------
    dict with keys:
        scl              : np.ndarray  SCL time series (µS)
        scr              : np.ndarray  Phasic SCR signal (µS)
        scr_events       : list[dict]  Detected SCR events (see detect_scr_events)
        scl_mean_us      : float  Mean SCL (µS) — baseline arousal level
        scl_std_us       : float  SCL std (µS) — slow arousal variability
        scl_min_us       : float
        scl_max_us       : float
        scr_count        : int    Number of detected SCR events
        scr_rate_per_min : float  SCR events per minute
        scr_mean_amplitude_us : float  Mean SCR amplitude (µS); NaN if 0 events
        scr_max_amplitude_us  : float  Max SCR amplitude (µS); NaN if 0 events
        duration_s       : float  Recording duration (s)
        n_samples        : int    Number of samples

    Raises
    ------
    ValueError  if gsr_us has fewer than 10 samples or fs ≤ 0.
    """
    gsr = np.asarray(gsr_us, dtype=float)
    if len(gsr) < 10:
        raise ValueError(f"Too short: need ≥ 10 samples, got {len(gsr)}")
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")

    duration_s = len(gsr) / fs

    scl = extract_scl(gsr, fs, lowpass_hz=lowpass_hz)
    scr = extract_scr(gsr, scl)
    events = detect_scr_events(
        scr, fs,
        min_amplitude_us=min_amplitude_us,
        min_rise_s=min_rise_s,
        max_rise_s=max_rise_s,
        min_interval_s=min_interval_s,
    )

    amps = [e["amplitude_us"] for e in events]
    scr_count = len(events)
    scr_rate  = scr_count / duration_s * 60.0

    return {
        "scl":               scl,
        "scr":               scr,
        "scr_events":        events,
        "scl_mean_us":       float(np.mean(scl)),
        "scl_std_us":        float(np.std(scl, ddof=1)) if len(scl) > 1 else 0.0,
        "scl_min_us":        float(np.min(scl)),
        "scl_max_us":        float(np.max(scl)),
        "scr_count":         scr_count,
        "scr_rate_per_min":  scr_rate,
        "scr_mean_amplitude_us": float(np.mean(amps)) if amps else float("nan"),
        "scr_max_amplitude_us":  float(np.max(amps))  if amps else float("nan"),
        "duration_s":        duration_s,
        "n_samples":         len(gsr),
    }
