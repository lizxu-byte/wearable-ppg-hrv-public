"""
synthetic.py — Synthetic PPG signal generator for algorithm validation

Generates realistic PPG-like signals with known ground-truth parameters
(exact peak indices, RR intervals, HR). Used to validate every processing
step with quantifiable metrics instead of visual inspection.

Waveform model: sum of three Gaussians per beat
  G1 — systolic peak   (dominant, ~15% into beat period)
  G2 — dicrotic notch  (small dip, ~50% into beat period; inverted)
  G3 — dicrotic wave   (secondary peak, ~65% into beat period)

This morphology is the minimum needed to challenge peak detectors that
might misfire on the dicrotic wave.

References:
    Elgendi M (2012). On the Analysis of Fingertip Photoplethysmogram
    Signals. Current Cardiology Reviews 8(1):14-25.
    https://doi.org/10.2174/157340312801215782
    (PPG waveform morphology: systolic peak, dicrotic notch, dicrotic wave)

    Charlton PH et al. (2020). Modeling arterial pulse waves in healthy aging:
    theoretical calculations and in vivo measurements. Am J Physiol Heart Circ
    Physiol 317:H144-H155. https://doi.org/10.1152/ajpheart.00218.2020
    (Parametric PPG/arterial waveform generation)

    Clifford GD et al. (2005). Synthetic ECG generation and Bayesian filtering
    using a Gaussian mixture model. Physiol Meas 26(6):707-729.
    https://doi.org/10.1088/0967-3334/26/6/014
    (Sum-of-Gaussians approach for physiological signal synthesis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Output container ───────────────────────────────────────────────────────────

@dataclass
class SyntheticPPG:
    """
    Complete synthetic PPG session with ground-truth labels.

    Attributes
    ----------
    signal       : np.ndarray (N,)  — composite signal (mV-like units)
    signal_clean : np.ndarray (N,)  — noise-free version (useful for RMSE)
    time_s       : np.ndarray (N,)  — time axis in seconds
    peak_indices : np.ndarray (K,)  — true systolic peak sample indices
    peak_times_s : np.ndarray (K,)  — true peak times (seconds)
    rr_ms        : np.ndarray (K-1,)— true RR intervals (ms)
    fs           : int              — sampling rate (Hz)
    params       : dict             — all generation parameters (for logging)
    """
    signal:       np.ndarray
    signal_clean: np.ndarray
    time_s:       np.ndarray
    peak_indices: np.ndarray
    peak_times_s: np.ndarray
    rr_ms:        np.ndarray
    fs:           int
    params:       dict = field(default_factory=dict)


# ── Core generator ─────────────────────────────────────────────────────────────

def generate(
    duration_s: float = 180.0,
    fs: int = 256,
    mean_hr_bpm: float = 70.0,
    hr_variability_ms: float = 30.0,
    noise_snr_db: float = 20.0,
    baseline_amplitude: float = 0.3,
    baseline_freq_hz: float = 0.15,
    add_motion_artifact: bool = False,
    motion_start_s: float = 60.0,
    motion_duration_s: float = 5.0,
    motion_amplitude: float = 2.0,
    seed: int = 42,
) -> SyntheticPPG:
    """
    Generate a synthetic PPG signal with known ground-truth peak positions.

    Waveform per beat (normalised to beat period T):
        systolic peak   at 0.15·T  σ=0.04·T  amplitude=1.0
        dicrotic notch  at 0.50·T  σ=0.03·T  amplitude=-0.15  (inverted)
        dicrotic wave   at 0.65·T  σ=0.05·T  amplitude=0.25

    Parameters
    ----------
    duration_s          : Total signal length (seconds).
    fs                  : Sampling rate (Hz). Default 256 (Shimmer3 nominal).
    mean_hr_bpm         : Mean heart rate (beats per minute).
    hr_variability_ms   : Std of beat-to-beat RR variation (ms).
                          Set to 0 for perfectly regular rhythm.
    noise_snr_db        : Signal-to-noise ratio for additive white Gaussian
                          noise (dB). Higher = cleaner signal.
    baseline_amplitude  : Peak amplitude of sinusoidal baseline drift
                          (same units as signal). 0 = no drift.
    baseline_freq_hz    : Frequency of baseline drift (Hz). Typical
                          respiratory rate: 0.1–0.4 Hz.
    add_motion_artifact : If True, inject a burst of band-limited noise
                          to simulate arm movement.
    motion_start_s      : Start time of motion burst (seconds).
    motion_duration_s   : Duration of motion burst (seconds).
    motion_amplitude    : Amplitude of motion burst (signal units).
    seed                : Random seed for reproducibility.

    Returns
    -------
    SyntheticPPG
        Signal arrays and ground-truth peak labels.
    """
    rng = np.random.default_rng(seed)
    n_total = int(duration_s * fs)
    t = np.arange(n_total) / fs  # seconds

    # ── Beat timing ───────────────────────────────────────────────────────────
    mean_rr_ms = 60_000.0 / mean_hr_bpm
    # Draw RR intervals from Gaussian; clip to physiological range 333–1500 ms
    rr_ms_sequence = []
    beat_starts_s  = []
    current_s = 0.05  # small offset so first beat isn't at t=0

    while current_s < duration_s:
        rr = rng.normal(mean_rr_ms, hr_variability_ms)
        rr = float(np.clip(rr, 333.0, 1500.0))
        beat_starts_s.append(current_s)
        rr_ms_sequence.append(rr)
        current_s += rr / 1000.0

    beat_starts_s  = np.array(beat_starts_s)
    rr_ms_sequence = np.array(rr_ms_sequence)

    # ── Build clean waveform ──────────────────────────────────────────────────
    clean = np.zeros(n_total)

    # Gaussian helper: add G(amplitude, center_s, sigma_s) to array
    def _add_gaussian(arr, amp, center_s, sigma_s):
        idx_lo = max(0, int((center_s - 4 * sigma_s) * fs))
        idx_hi = min(n_total, int((center_s + 4 * sigma_s) * fs) + 1)
        if idx_lo >= idx_hi:
            return
        t_seg = t[idx_lo:idx_hi]
        arr[idx_lo:idx_hi] += amp * np.exp(-0.5 * ((t_seg - center_s) / sigma_s) ** 2)

    # Waveform fractions (relative to beat period T)
    # Ref: Elgendi (2012) — systolic peak leads, dicrotic notch ~50%, dicrotic wave ~65%
    SYS_FRAC    = 0.15   # systolic peak position fraction
    NOTCH_FRAC  = 0.50   # dicrotic notch position
    DIC_FRAC    = 0.65   # dicrotic wave position

    SYS_SIG_FRAC    = 0.04
    NOTCH_SIG_FRAC  = 0.03
    DIC_SIG_FRAC    = 0.05

    SYS_AMP     =  1.00
    NOTCH_AMP   = -0.15
    DIC_AMP     =  0.25

    true_peak_indices = []

    for i, (t_start, rr) in enumerate(zip(beat_starts_s, rr_ms_sequence)):
        T = rr / 1000.0  # beat period in seconds

        t_sys   = t_start + SYS_FRAC   * T
        t_notch = t_start + NOTCH_FRAC * T
        t_dic   = t_start + DIC_FRAC   * T

        _add_gaussian(clean, SYS_AMP,   t_sys,   SYS_SIG_FRAC   * T)
        _add_gaussian(clean, NOTCH_AMP, t_notch, NOTCH_SIG_FRAC * T)
        _add_gaussian(clean, DIC_AMP,   t_dic,   DIC_SIG_FRAC   * T)

        peak_idx = int(round(t_sys * fs))
        if 0 <= peak_idx < n_total:
            true_peak_indices.append(peak_idx)

    true_peak_indices = np.array(true_peak_indices, dtype=int)
    true_peak_times_s = true_peak_indices / fs
    true_rr_ms = np.diff(true_peak_times_s) * 1000.0

    # ── Baseline drift ────────────────────────────────────────────────────────
    # Sinusoidal respiratory baseline (Charlton 2020)
    baseline = baseline_amplitude * np.sin(2 * np.pi * baseline_freq_hz * t)

    # ── AWGN ─────────────────────────────────────────────────────────────────
    signal_power = np.mean(clean ** 2)
    if signal_power > 0 and not np.isinf(noise_snr_db):
        noise_power = signal_power / (10 ** (noise_snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), n_total)
    else:
        noise = np.zeros(n_total)

    # ── Motion artifact ───────────────────────────────────────────────────────
    motion = np.zeros(n_total)
    if add_motion_artifact:
        idx_start = int(motion_start_s * fs)
        idx_end   = min(n_total, idx_start + int(motion_duration_s * fs))
        n_motion  = idx_end - idx_start
        # Band-limited noise 1–10 Hz (typical movement frequency)
        raw_noise = rng.normal(0, 1, n_motion)
        # Simple moving-average bandpass approximation
        k_lo = max(1, int(fs / 10))  # ~10 Hz low-pass kernel
        kernel = np.ones(k_lo) / k_lo
        bandlimited = np.convolve(raw_noise, kernel, mode="same")
        bandlimited *= motion_amplitude / (np.std(bandlimited) + 1e-9)
        motion[idx_start:idx_end] = bandlimited

    # ── Composite signal ──────────────────────────────────────────────────────
    composite = clean + baseline + noise + motion

    params = {
        "duration_s":          duration_s,
        "fs":                  fs,
        "mean_hr_bpm":         mean_hr_bpm,
        "hr_variability_ms":   hr_variability_ms,
        "noise_snr_db":        noise_snr_db,
        "baseline_amplitude":  baseline_amplitude,
        "baseline_freq_hz":    baseline_freq_hz,
        "add_motion_artifact": add_motion_artifact,
        "motion_start_s":      motion_start_s,
        "motion_duration_s":   motion_duration_s,
        "motion_amplitude":    motion_amplitude,
        "seed":                seed,
        "n_beats":             len(true_peak_indices),
        "actual_mean_rr_ms":   float(np.mean(true_rr_ms)) if len(true_rr_ms) > 0 else float("nan"),
    }

    return SyntheticPPG(
        signal=composite,
        signal_clean=clean,
        time_s=t,
        peak_indices=true_peak_indices,
        peak_times_s=true_peak_times_s,
        rr_ms=true_rr_ms,
        fs=fs,
        params=params,
    )


# ── Convenience presets ────────────────────────────────────────────────────────

def make_clean(hr_bpm: float = 70.0, duration_s: float = 60.0,
               fs: int = 256, seed: int = 42) -> SyntheticPPG:
    """Clean signal: no noise, no drift. For filter frequency-response tests."""
    return generate(duration_s=duration_s, fs=fs, mean_hr_bpm=hr_bpm,
                    hr_variability_ms=0.0, noise_snr_db=float("inf"),
                    baseline_amplitude=0.0, add_motion_artifact=False, seed=seed)


def make_noisy(hr_bpm: float = 70.0, snr_db: float = 15.0,
               duration_s: float = 60.0, fs: int = 256,
               seed: int = 42) -> SyntheticPPG:
    """Noisy signal with baseline drift. Standard test for filtering."""
    return generate(duration_s=duration_s, fs=fs, mean_hr_bpm=hr_bpm,
                    hr_variability_ms=30.0, noise_snr_db=snr_db,
                    baseline_amplitude=0.3, baseline_freq_hz=0.15,
                    add_motion_artifact=False, seed=seed)


def make_with_motion(hr_bpm: float = 70.0, duration_s: float = 120.0,
                     fs: int = 256, seed: int = 42) -> SyntheticPPG:
    """Signal with motion artifact burst at 60–65 s."""
    return generate(duration_s=duration_s, fs=fs, mean_hr_bpm=hr_bpm,
                    hr_variability_ms=30.0, noise_snr_db=20.0,
                    baseline_amplitude=0.3, baseline_freq_hz=0.15,
                    add_motion_artifact=True,
                    motion_start_s=60.0, motion_duration_s=5.0,
                    motion_amplitude=2.0, seed=seed)
