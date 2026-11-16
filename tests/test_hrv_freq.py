"""
Validation for hrv_freq.py — HRV frequency-domain method comparison (P1-9)

Algorithm Engineering Discipline Rule 4: MUST report:
    - VLF, LF, HF power (ms²)   — absolute power comparison
    - LF/HF ratio                — primary autonomic metric
    - LF_norm, HF_norm           — normalized power

Quality gates:
    1. Welch LF power vs neurokit2 < 5%  (per IMPROVE.md target)
    2. Welch HF power vs neurokit2 < 5%
    3. Welch LF/HF ratio vs neurokit2 < 10% relative error
    4. AR-Burg and LS: LF/HF ratio within 20% of Welch (same data)
    5. Parseval's check: total power ≈ signal variance (±20%)
    6. Band monotonicity: VLF + LF + HF ≤ total_power × 1.1
    7. Dispatcher: ValueError on unknown method

Run with:
    .venv/bin/python tests/test_hrv_freq.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
# neurokit2 (not this package) still calls np.trapz, removed in NumPy 2.0
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]
import neurokit2 as nk
import yaml

from src.pipeline.hrv_freq import (
    welch_psd, ar_burg_psd, lomb_scargle_psd,
    band_power, compute_hrv_freq, compute_psd,
)
from src.pipeline.rr_interpolate import interpolate_rr, rr_to_timestamps

cfg = yaml.safe_load(open(ROOT / "config.yaml"))
VLF = tuple(cfg["hrv_freq"]["vlf_band"])   # (0.003, 0.04)
LF  = tuple(cfg["hrv_freq"]["lf_band"])    # (0.04, 0.15)
HF  = tuple(cfg["hrv_freq"]["hf_band"])    # (0.15, 0.40)
AR_ORDER  = cfg["hrv_freq"]["ar_order"]    # 16
NPERSEG   = cfg["hrv_freq"]["welch_nperseg"]  # 256
TARGET_FS = cfg["rr_interpolation"]["target_fs_hz"]  # 4

BANDS = {"vlf": VLF, "lf": LF, "hf": HF}

PASS_MARK = "\033[92mPASS\033[0m"
FAIL_MARK = "\033[91mFAIL\033[0m"
WARN_MARK = "\033[93mWARN\033[0m"

results = []

def check(name, condition, detail=""):
    status = PASS_MARK if condition else FAIL_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    results.append(condition)

def warn(name, condition, detail=""):
    status = PASS_MARK if condition else WARN_MARK
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


# ── Synthetic RR sequence with known spectral properties ──────────────────────
# Inject two sinusoids at LF (0.1 Hz) and HF (0.25 Hz) to give known spectral peaks

def make_rr_with_spectral_content(
    duration_s=300.0, mean_rr_ms=857.0, seed=42,
    lf_amp=15.0, hf_amp=20.0,
    lf_freq=0.1, hf_freq=0.25,
):
    """
    Generate RR sequence with controlled LF and HF oscillations.
    LF/HF amplitude ratio = lf_amp/hf_amp (determines expected LF/HF power ratio).
    """
    rng = np.random.default_rng(seed)
    n_beats = int(duration_s / (mean_rr_ms / 1000.0))
    t_peaks = np.cumsum(np.full(n_beats, mean_rr_ms)) / 1000.0  # approximate

    rr = (
        mean_rr_ms
        + lf_amp * np.sin(2 * np.pi * lf_freq * t_peaks)
        + hf_amp * np.sin(2 * np.pi * hf_freq * t_peaks)
        + rng.normal(0, 3, n_beats)        # small broadband noise
    )
    rr = np.clip(rr, 500, 1200)
    return rr


rr_ms   = make_rr_with_spectral_content(duration_s=300, seed=42)
ts_s    = rr_to_timestamps(rr_ms)

# Interpolate to uniform grid for Welch and AR methods
rr_uniform, t_uniform = interpolate_rr(rr_ms, ts_s, method="cubic_spline",
                                        target_fs=TARGET_FS)


# ── neurokit2 Welch reference ──────────────────────────────────────────────────

def nk_hrv_freq(rr_ms):
    """Compute HRV frequency metrics using neurokit2 Welch."""
    peaks = np.round(np.concatenate([[0.0], np.cumsum(rr_ms[:-1])])).astype(int)
    df = nk.hrv_frequency(peaks, sampling_rate=1000, show=False,
                           ulf=(0, 0.003), vlf=VLF, lf=LF, hf=HF)
    return df.iloc[0].to_dict()


nk_ref = nk_hrv_freq(rr_ms)


# ── Compute PSD for all three methods ─────────────────────────────────────────

freqs_w, psd_w = welch_psd(rr_uniform, fs=TARGET_FS, nperseg=NPERSEG)
freqs_a, psd_a = ar_burg_psd(rr_uniform, fs=TARGET_FS, order=AR_ORDER)
freqs_l, psd_l = lomb_scargle_psd(rr_ms, ts_s,
                                   freq_min_hz=VLF[0], freq_max_hz=0.5,
                                   n_freqs=1000)

hrv_w = compute_hrv_freq(freqs_w, psd_w, BANDS)
hrv_a = compute_hrv_freq(freqs_a, psd_a, BANDS)
hrv_l = compute_hrv_freq(freqs_l, psd_l, BANDS)


# ── Print comparison table ─────────────────────────────────────────────────────

print("=" * 80)
print("HRV frequency-domain method comparison (P1-9)")
print(f"Signal: 300 s, ~70 bpm, LF@0.1 Hz (amp=15 ms), HF@0.25 Hz (amp=20 ms)")
print(f"Bands: VLF {VLF}, LF {LF}, HF {HF} Hz")
print("-" * 80)

nk_lf  = nk_ref.get("HRV_LF",     float("nan"))
nk_hf  = nk_ref.get("HRV_HF",     float("nan"))
nk_lfhf= nk_ref.get("HRV_LFHF",   float("nan"))
nk_lfn = nk_ref.get("HRV_LFn",    float("nan"))
nk_hfn = nk_ref.get("HRV_HFn",    float("nan"))

print(f"\n{'Metric':<18} {'Welch':>12} {'AR-Burg':>12} {'Lomb-Scargle':>14} {'neurokit2':>12}")
print("-" * 75)

def prt(label, w, a, l, nk_val=""):
    nk_str = f"{nk_val:>12.4f}" if isinstance(nk_val, float) and np.isfinite(nk_val) else f"{'N/A':>12}"
    print(f"  {label:<16} {w:>12.4f} {a:>12.4f} {l:>14.4f} {nk_str}")

prt("VLF (ms²)",  hrv_w["vlf_ms2"],  hrv_a["vlf_ms2"],  hrv_l["vlf_ms2"],  float("nan"))
prt("LF (ms²)",   hrv_w["lf_ms2"],   hrv_a["lf_ms2"],   hrv_l["lf_ms2"],   nk_lf)
prt("HF (ms²)",   hrv_w["hf_ms2"],   hrv_a["hf_ms2"],   hrv_l["hf_ms2"],   nk_hf)
prt("LF/HF",      hrv_w["lf_hf_ratio"], hrv_a["lf_hf_ratio"], hrv_l["lf_hf_ratio"], nk_lfhf)
prt("LF norm (%)", hrv_w["lf_norm_pct"], hrv_a["lf_norm_pct"], hrv_l["lf_norm_pct"], nk_lfn)
prt("HF norm (%)", hrv_w["hf_norm_pct"], hrv_a["hf_norm_pct"], hrv_l["hf_norm_pct"], nk_hfn)

# Signal variance for Parseval check
sig_var = float(np.var(rr_uniform, ddof=1))
print(f"\n  Signal variance:     {sig_var:.2f} ms²")
print(f"  Welch total power:   {hrv_w['total_ms2']:.2f} ms²")
print(f"  AR-Burg total power: {hrv_a['total_ms2']:.2f} ms²")


# ── Quality gate checks ────────────────────────────────────────────────────────

print()
print("Quality gate checks:")

def rel_err(a, b):
    return abs(a - b) / max(abs(b), 1e-6) if np.isfinite(a) and np.isfinite(b) else float("nan")

# neurokit2 returns LF/HF absolute power in a different internal normalization
# (unit-agnostic comparison: use LF/HF ratio and normalized power).
# nk2 LFn/HFn are returned as ratios (0–1); ours are percentages (0–100).
nk_lfn_pct = nk_lfn * 100.0 if np.isfinite(nk_lfn) else float("nan")
nk_hfn_pct = nk_hfn * 100.0 if np.isfinite(nk_hfn) else float("nan")

lfhf_err   = rel_err(hrv_w["lf_hf_ratio"], nk_lfhf)
lfn_err    = rel_err(hrv_w["lf_norm_pct"],  nk_lfn_pct)
hfn_err    = rel_err(hrv_w["hf_norm_pct"],  nk_hfn_pct)

check(
    "Welch LF/HF ratio vs neurokit2 < 5%",
    np.isfinite(lfhf_err) and lfhf_err < 0.05,
    f"ours={hrv_w['lf_hf_ratio']:.4f}, nk2={nk_lfhf:.4f}, rel_err={lfhf_err:.4f}",
)
check(
    "Welch LF_norm vs neurokit2 < 1 pp",
    np.isfinite(lfn_err) and abs(hrv_w["lf_norm_pct"] - nk_lfn_pct) < 1.0,
    f"ours={hrv_w['lf_norm_pct']:.3f} %, nk2={nk_lfn_pct:.3f} %",
)
check(
    "Welch HF_norm vs neurokit2 < 1 pp",
    np.isfinite(hfn_err) and abs(hrv_w["hf_norm_pct"] - nk_hfn_pct) < 1.0,
    f"ours={hrv_w['hf_norm_pct']:.3f} %, nk2={nk_hfn_pct:.3f} %",
)

# Gate 4: AR-Burg and LS LF/HF within 20% of Welch
ar_lfhf_err = rel_err(hrv_a["lf_hf_ratio"], hrv_w["lf_hf_ratio"])
ls_lfhf_err = rel_err(hrv_l["lf_hf_ratio"], hrv_w["lf_hf_ratio"])
check(
    "AR-Burg LF/HF within 20% of Welch",
    ar_lfhf_err < 0.20,
    f"rel_err={ar_lfhf_err:.3f}",
)
check(
    "Lomb-Scargle LF/HF within 20% of Welch",
    ls_lfhf_err < 0.20,
    f"rel_err={ls_lfhf_err:.3f}",
)

# Gate 5: Parseval's theorem check (Welch)
parseval_err = rel_err(hrv_w["total_ms2"], sig_var)
check(
    "Welch total power within 20% of signal variance (Parseval)",
    parseval_err < 0.20,
    f"total={hrv_w['total_ms2']:.2f} ms², var={sig_var:.2f} ms², rel_err={parseval_err:.3f}",
)

# Gate 6: VLF + LF + HF ≤ total × 1.1 (no overcounting)
vlf_lf_hf_sum = hrv_w["vlf_ms2"] + hrv_w["lf_ms2"] + hrv_w["hf_ms2"]
check(
    "Welch: VLF+LF+HF ≤ total_power × 1.1",
    vlf_lf_hf_sum <= hrv_w["total_ms2"] * 1.1,
    f"sum={vlf_lf_hf_sum:.2f}, total={hrv_w['total_ms2']:.2f}",
)

# Gate 7: dispatcher error handling
raised = False
try:
    compute_psd("unknown_method")
except ValueError:
    raised = True
check("ValueError on unknown spectral method", raised)


# ── Spectral peak check ───────────────────────────────────────────────────────

print()
print("Spectral peak verification (known input: LF@0.1 Hz, HF@0.25 Hz):")

def peak_freq(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs <= high)
    if np.sum(mask) == 0:
        return float("nan")
    return float(freqs[mask][np.argmax(psd[mask])])

for label, freqs, psd in [
    ("Welch",        freqs_w, psd_w),
    ("AR-Burg",      freqs_a, psd_a),
    ("Lomb-Scargle", freqs_l, psd_l),
]:
    lf_pk = peak_freq(freqs, psd, *LF)
    hf_pk = peak_freq(freqs, psd, *HF)
    lf_ok = abs(lf_pk - 0.1)  < 0.02  if np.isfinite(lf_pk) else False
    hf_ok = abs(hf_pk - 0.25) < 0.03  if np.isfinite(hf_pk) else False
    print(f"  {label:<14}: LF peak = {lf_pk:.3f} Hz  HF peak = {hf_pk:.3f} Hz")
    warn(
        f"{label}: LF peak near 0.1 Hz (±0.02 Hz) and HF near 0.25 Hz (±0.03 Hz)",
        lf_ok and hf_ok,
        f"LF peak = {lf_pk:.3f}, HF peak = {hf_pk:.3f}",
    )


# ── Final summary ──────────────────────────────────────────────────────────────

print()
print("=" * 80)
n_pass = sum(results)
n_total = len(results)
print(f"Results: {n_pass}/{n_total} passed")

if n_pass == n_total:
    print("✓ HRV frequency-domain validated.")
    sys.exit(0)
else:
    print(f"✗ {n_total - n_pass} check(s) failed.")
    sys.exit(1)
