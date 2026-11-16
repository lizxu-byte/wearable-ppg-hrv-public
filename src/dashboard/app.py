"""
app.py — PPG Algorithm Comparison Dashboard (P4-1 through P4-4)

P4-1: subject/session selector, PPG raw + filtered waveforms, window navigation.
P4-2: filter preset / RR-cleaning selectors, HRV metrics card, RR chart.
P4-3: HRV panel — PSD (Welch/AR-Burg/LS), Poincaré plot, time-domain table.
P4-4: cross-record comparison — batch compute every listed record, sortable /
      filterable DataTable, RMSSD vs HR scatter.

Data sources (any subset may be present):
  * Local Shimmer3 CSV sessions under data.root_dir — never shipped with the
    repository; session labels are neutral indices, not file names.
  * BIDMC PPG and Respiration Dataset (PhysioNet) under data/bidmc —
    fetch with scripts/download_bidmc.py.
  * Synthetic PPG presets from src/validation/synthetic.py (always available).

Layout: sidebar (all pipeline controls) + tabbed main area.
  Tab 1 "Session Analysis"  — P4-1 to P4-3 content.
  Tab 2 "Cross-Subject"     — P4-4 batch table + scatter.

Cache: server-side dict keyed by "{source_key}::{filter_key}::{peak_method}".
       RR cleaning and HRV metrics run inside callbacks (~5 ms each).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from dash import Dash, Input, Output, State, dash_table, dcc, html

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.io.shimmer_reader import load_session, resolve_data_root
from src.pipeline.filter import bandpass_butterworth
from src.pipeline.hrv_freq import compute_hrv_freq, compute_psd
from src.pipeline.hrv_nonlinear import compute_hrv_nonlinear
from src.pipeline.hrv_time import compute_hrv_time
from src.pipeline.peaks import detect_peaks
from src.pipeline.rr_clean import clean_rr
from src.pipeline.rr_interpolate import interpolate_rr, rr_to_timestamps
from src.validation.public_dataset import load_bidmc_record
from src.validation.synthetic import generate as generate_synthetic


# ── Config ─────────────────────────────────────────────────────────────────────

with open(ROOT / "config.yaml", encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)

DATA_ROOT    = resolve_data_root(CFG)
BIDMC_DIR    = ROOT / CFG["validation"]["public_dataset_dir"]
FS_NOMINAL   = int(CFG["sampling"]["fs"])
WINDOW_S     = float(CFG["dashboard"]["default_window_s"])
HOST         = CFG["dashboard"]["host"]
PORT         = int(CFG["dashboard"]["port"])
DEBUG        = bool(CFG["dashboard"]["debug"])
HRV_BANDS    = {
    "vlf": tuple(CFG["hrv_freq"]["vlf_band"]),
    "lf":  tuple(CFG["hrv_freq"]["lf_band"]),
    "hf":  tuple(CFG["hrv_freq"]["hf_band"]),
}
WELCH_NPERSEG = int(CFG["hrv_freq"]["welch_nperseg"])
AR_ORDER      = int(CFG["hrv_freq"]["ar_order"])
TARGET_FS     = float(CFG["rr_interpolation"]["target_fs_hz"])


# ── Method menus ───────────────────────────────────────────────────────────────

FILTER_PRESETS: dict[str, tuple[str, float, float, int]] = {
    "0.4-10-o4": ("0.4–10 Hz, order 4  ✦ recommended", 0.4, 10.0, 4),
    "0.5-8-o4":  ("0.5–8 Hz, order 4   Elgendi 2013",  0.5,  8.0, 4),
    "0.5-5-o4":  ("0.5–5 Hz, order 4   strict HRV",    0.5,  5.0, 4),
}
FILTER_DEFAULT = "0.4-10-o4"

PEAK_OPTIONS  = [
    {"label": "Elgendi 2013  ✦ recommended", "value": "elgendi"},
    {"label": "Prominence (scipy)",           "value": "prominence"},
]
PEAK_DEFAULT  = "elgendi"

CLEAN_OPTIONS = [
    {"label": "Malik 1996  ✦ recommended",  "value": "malik"},
    {"label": "Physiological + 3SD",         "value": "physiological_3sd"},
    {"label": "Karlsson 2012",               "value": "karlsson"},
]
CLEAN_DEFAULT = "malik"

PSD_OPTIONS   = [
    {"label": "Welch (interpolated)  ✦ recommended", "value": "welch"},
    {"label": "AR-Burg (parametric)",                 "value": "ar_burg"},
    {"label": "Lomb-Scargle (no interpolation)",      "value": "lomb_scargle"},
]
PSD_DEFAULT   = "welch"


# ── Server-side cache ──────────────────────────────────────────────────────────

_SESSION_CACHE: dict[str, dict] = {}


# Demo signals for the public build. Parameters are generate() keyword
# arguments; the motion preset reproduces make_with_motion() (burst at 60–65 s).
SYNTHETIC_PRESETS: dict[str, tuple[str, dict]] = {
    "clean":  ("Synthetic · 70 bpm, SNR 20 dB", dict(duration_s=180.0, mean_hr_bpm=70.0)),
    "noisy":  ("Synthetic · 70 bpm, SNR 5 dB",  dict(duration_s=180.0, mean_hr_bpm=70.0,
                                                    noise_snr_db=5.0)),
    "motion": ("Synthetic · motion burst 60–65 s", dict(duration_s=120.0, mean_hr_bpm=70.0,
                                                       add_motion_artifact=True)),
}


def _load_signal(source_key: str) -> dict:
    """Load raw PPG for a source key of the form '<kind>:<locator>'."""
    kind, _, loc = source_key.partition(":")
    if kind == "shimmer":
        session = load_session(Path(loc))
        fs = round(session.fs_measured) if np.isfinite(session.fs_measured) else FS_NOMINAL
        return {"time_s": session.time_s, "ppg": session.ppg, "fs": fs, "unit": "mV",
                "duration_s": session.duration_s, "n_samples": session.n_samples}
    if kind == "bidmc":
        rec = load_bidmc_record(Path(loc), detect_peaks=False)
        return {"time_s": np.arange(rec.n_samples) / rec.fs, "ppg": rec.ppg,
                "fs": rec.fs, "unit": "a.u.",
                "duration_s": rec.duration_s, "n_samples": rec.n_samples}
    if kind == "synthetic":
        sig = generate_synthetic(**SYNTHETIC_PRESETS[loc][1])
        return {"time_s": sig.time_s, "ppg": sig.signal, "fs": sig.fs, "unit": "a.u.",
                "duration_s": float(sig.time_s[-1] - sig.time_s[0]),
                "n_samples": len(sig.signal)}
    raise ValueError(f"Unknown source: {source_key}")


def _load_and_cache(source_key: str, filter_key: str, peak_method: str) -> dict:
    key = f"{source_key}::{filter_key}::{peak_method}"
    if key in _SESSION_CACHE:
        return _SESSION_CACHE[key]
    sig = _load_signal(source_key)
    fs = sig["fs"]
    _, lo, hi, order = FILTER_PRESETS[filter_key]
    ppg_filt = bandpass_butterworth(sig["ppg"], fs, lowcut_hz=lo, highcut_hz=hi, order=order)
    try:
        peaks = detect_peaks(ppg_filt, fs, method=peak_method)
    except Exception:
        peaks = np.array([], dtype=int)
    _SESSION_CACHE[key] = {
        "time_s": sig["time_s"], "ppg_raw": sig["ppg"], "ppg_filt": ppg_filt,
        "peaks": peaks, "fs": fs, "unit": sig["unit"],
        "duration_s": sig["duration_s"], "n_samples": sig["n_samples"],
    }
    return _SESSION_CACHE[key]


def _compute_rr_metrics(data: dict, clean_method: str) -> dict:
    peaks  = data["peaks"]
    fs     = data["fs"]
    rr_raw = np.diff(peaks) / fs * 1000.0
    if len(rr_raw) < 2:
        return {"error": "Too few peaks"}
    rr_clean, mask = clean_rr(rr_raw, method=clean_method)
    n_removed = int(np.sum(~mask))
    try:
        hrv = compute_hrv_time(rr_clean)
    except Exception as exc:
        hrv = {"error": str(exc)}
    hrv.update({
        "n_peaks": len(peaks), "n_rr_raw": len(rr_raw),
        "n_rr_clean": len(rr_clean), "n_rr_removed": n_removed,
        "removal_pct": 100.0 * n_removed / len(rr_raw) if len(rr_raw) > 0 else 0.0,
        "peak_times_s": data["time_s"][peaks],
        "rr_raw_ms": rr_raw, "rr_clean_ms": rr_clean, "rr_mask": mask,
    })
    return hrv


def _compute_hrv_freq_metrics(rr_clean: np.ndarray, psd_method: str) -> dict:
    if len(rr_clean) < 4:
        return {"error": "Too few RR"}
    ts = rr_to_timestamps(rr_clean)
    try:
        if psd_method in ("welch", "ar_burg"):
            rr_u, _ = interpolate_rr(rr_clean, ts, method="cubic_spline", target_fs=TARGET_FS)
            freqs, psd = compute_psd(psd_method, rr_uniform_ms=rr_u, fs=TARGET_FS)
        else:
            freqs, psd = compute_psd(psd_method, rr_ms=rr_clean, timestamps_s=ts)
    except Exception as exc:
        return {"error": str(exc)}
    try:
        bp = compute_hrv_freq(freqs, psd, HRV_BANDS)
    except Exception as exc:
        bp = {"error": str(exc)}
    return {"freqs": freqs, "psd": psd, "band_powers": bp}


# ── Subject / session discovery ────────────────────────────────────────────────

def _list_subjects(include_synthetic: bool = True) -> list[dict]:
    subjects = [{"label": f"Shimmer subject {i} (local)", "value": f"shimmer:{i}"}
                for i in range(67) if (DATA_ROOT / str(i)).is_dir()]
    subjects += [{"label": f"BIDMC {p.stem}", "value": f"bidmc:{p.stem}"}
                 for p in sorted(BIDMC_DIR.glob("bidmc[0-9][0-9].hea"))]
    if include_synthetic:
        subjects.append({"label": "Synthetic signals", "value": "synthetic"})
    return subjects


def _list_sessions(subject: str) -> list[dict]:
    kind, _, loc = subject.partition(":")
    if kind == "shimmer":
        # Index labels only: Shimmer file names embed participant initials.
        csvs = sorted((DATA_ROOT / loc).rglob("*.csv"))
        return [{"label": f"Session {k}", "value": f"shimmer:{p}"}
                for k, p in enumerate(csvs, start=1)]
    if kind == "bidmc":
        return [{"label": f"{loc} · 8 min ICU recording", "value": f"bidmc:{BIDMC_DIR / loc}"}]
    if kind == "synthetic":
        return [{"label": label, "value": f"synthetic:{k}"}
                for k, (label, _) in SYNTHETIC_PRESETS.items()]
    return []


def _session_label(source_key: str) -> str:
    kind, _, loc = source_key.partition(":")
    if kind == "synthetic":
        return SYNTHETIC_PRESETS[loc][0]
    if kind == "bidmc":
        return f"BIDMC {Path(loc).name}"
    return "Shimmer session (local)"


def _batch_compute(filter_key: str, peak_method: str,
                   clean_method: str, psd_method: str) -> list[dict]:
    """Compute HRV metrics for the first session of every real (non-synthetic) record."""
    results = []
    for subj in _list_subjects(include_synthetic=False):
        sid      = subj["value"]
        sessions = _list_sessions(sid)
        row: dict = {"subject": subj["label"], "short": sid.partition(":")[2]}

        if not sessions:
            row["status"] = "no sessions"
            results.append(row)
            continue

        source_key = sessions[0]["value"]
        row["session"] = sessions[0]["label"]

        try:
            data = _load_and_cache(source_key, filter_key, peak_method)
            row["duration_s"] = round(data["duration_s"], 1)
            row["n_peaks"]    = len(data["peaks"])

            m = _compute_rr_metrics(data, clean_method)
            if "error" in m:
                row["status"] = m["error"]
                results.append(row)
                continue

            row["hr_mean"]     = _r(m.get("mean_hr_bpm"))
            row["rmssd_ms"]    = _r(m.get("rmssd_ms"))
            row["sdnn_ms"]     = _r(m.get("sdnn_ms"))
            row["pnn50_pct"]   = _r(m.get("pnn50_pct"))
            row["n_rr_clean"]  = m["n_rr_clean"]
            row["removal_pct"] = _r(m.get("removal_pct"))

            try:
                nl = compute_hrv_nonlinear(m["rr_clean_ms"])
                row["sd1_ms"] = _r(nl["sd1_ms"])
                row["sd2_ms"] = _r(nl["sd2_ms"])
            except Exception:
                row["sd1_ms"] = row["sd2_ms"] = None

            freq_res = _compute_hrv_freq_metrics(m["rr_clean_ms"], psd_method)
            if "error" not in freq_res:
                lf_hf = freq_res["band_powers"].get("lf_hf_ratio")
                row["lf_hf"] = round(float(lf_hf), 2) if lf_hf and np.isfinite(lf_hf) else None
            else:
                row["lf_hf"] = None

            row["status"] = "OK"

        except Exception as exc:
            row["status"] = str(exc)[:60]

        results.append(row)
    return results


def _r(v, d=1):
    """Round to d decimals; return None for non-finite."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), d)


# ── App ────────────────────────────────────────────────────────────────────────

SUBJECTS = _list_subjects()
N_BATCH  = len(_list_subjects(include_synthetic=False))

app = Dash(__name__, title="PPG Comparison Dashboard",
           suppress_callback_exceptions=True)

# ── Style helpers ──────────────────────────────────────────────────────────────

_LABEL = dict(fontSize="11px", fontWeight="600", color="#64748b",
              letterSpacing="0.04em", textTransform="uppercase",
              display="block", marginBottom="5px")
_SDIV  = dict(borderTop="1px solid #e2e8f0", paddingTop="14px", marginTop="2px")
_CARD  = dict(backgroundColor="white", borderRadius="8px",
              border="1px solid #e2e8f0", padding="14px 16px")
_DD    = dict(fontSize="12px")
_MBOX  = dict(display="flex", flexDirection="column", alignItems="center",
              justifyContent="center", padding="10px 8px",
              backgroundColor="#f8fafc", border="1px solid #e2e8f0",
              borderRadius="6px", flex="1", minWidth="90px")


def _lbl(text):
    return html.Label(text, style=_LABEL)


def _sec(title, *children):
    return html.Div(style=_SDIV, children=[_lbl(title), *children])


def _metric(value, unit, sub, color="#1e293b"):
    return html.Div(style=_MBOX, children=[
        html.Span(value, style=dict(fontSize="20px", fontWeight="700",
                                    color=color, lineHeight="1")),
        html.Span(unit,  style=dict(fontSize="10px", color="#94a3b8", marginTop="2px")),
        html.Span(sub,   style=dict(fontSize="10px", fontWeight="600", color="#64748b",
                                    textTransform="uppercase", marginTop="4px",
                                    letterSpacing="0.04em")),
    ])


def _trow(label, value, unit="", hi=False):
    bg = "#f0f9ff" if hi else "white"
    return html.Tr(style=dict(backgroundColor=bg), children=[
        html.Td(label, style=dict(padding="5px 10px", fontSize="12px",
                                   color="#475569", width="160px")),
        html.Td(value, style=dict(padding="5px 10px", fontSize="13px",
                                   fontWeight="600", color="#1e293b",
                                   textAlign="right")),
        html.Td(unit,  style=dict(padding="5px 10px", fontSize="11px",
                                   color="#94a3b8", width="40px")),
    ])


# ── DataTable column spec ──────────────────────────────────────────────────────

_TABLE_COLS = [
    {"name": "Record",     "id": "subject",     "type": "text"},
    {"name": "Session",    "id": "session",     "type": "text"},
    {"name": "Dur (s)",    "id": "duration_s",  "type": "numeric"},
    {"name": "HR (bpm)",   "id": "hr_mean",     "type": "numeric"},
    {"name": "RMSSD (ms)", "id": "rmssd_ms",    "type": "numeric"},
    {"name": "SDNN (ms)",  "id": "sdnn_ms",     "type": "numeric"},
    {"name": "pNN50 (%)",  "id": "pnn50_pct",   "type": "numeric"},
    {"name": "SD1 (ms)",   "id": "sd1_ms",      "type": "numeric"},
    {"name": "SD2 (ms)",   "id": "sd2_ms",      "type": "numeric"},
    {"name": "LF/HF",      "id": "lf_hf",       "type": "numeric"},
    {"name": "Peaks",      "id": "n_peaks",     "type": "numeric"},
    {"name": "RR kept",    "id": "n_rr_clean",  "type": "numeric"},
    {"name": "Removed%",   "id": "removal_pct", "type": "numeric"},
    {"name": "Status",     "id": "status",      "type": "text"},
]

# ── BIDMC validation results (P3-2, runner.py, 53 ICU subjects) ───────────────
# Pre-computed: Butterworth 0.4-10 Hz order 4 (fixed) × 2 peak methods × 3 RR
# cleaners = 6 combinations. Values: median across 53 records (non-failed only).

_BIDMC_RESULTS = [
    # (peak, clean, hr_mae_med, rmssd_err_med, f1_med)   best → lowest hr_mae
    ("elgendi",    "malik",             4.97, 24.44, 0.946),
    ("elgendi",    "karlsson",          5.23, 25.93, 0.946),
    ("elgendi",    "physiological_3sd", 5.74, 36.48, 0.946),
    ("prominence", "malik",             5.71, 26.12, 0.944),
    ("prominence", "karlsson",          5.97, 27.40, 0.944),
    ("prominence", "physiological_3sd", 6.31, 37.89, 0.944),
]
_BIDMC_BEST = ("elgendi", "malik")


def _build_validation_tab() -> html.Div:
    """Static BIDMC validation summary — no callback needed."""
    cols = ["Pipeline", "HR MAE (bpm)", "RMSSD Δ (ms)", "F1"]
    rows_data = []
    for pk, cl, hr, rmssd, f1 in _BIDMC_RESULTS:
        is_best = (pk, cl) == _BIDMC_BEST
        label = f"{'★ ' if is_best else ''}{pk} + {cl}"
        rows_data.append((label, hr, rmssd, f1, is_best))

    def _cell(val, bold=False, color="#334155", align="right"):
        return html.Td(val, style=dict(
            padding="7px 14px", fontSize="13px",
            fontWeight="700" if bold else "400",
            color=color, textAlign=align,
            borderBottom="1px solid #f1f5f9"))

    table_rows = []
    for label, hr, rmssd, f1, best in rows_data:
        bg = "#eff6ff" if best else "white"
        table_rows.append(html.Tr(style=dict(backgroundColor=bg), children=[
            _cell(label, bold=best, color="#1e293b" if best else "#334155",
                  align="left"),
            _cell(f"{hr:.2f}",    bold=best, color="#dc2626" if hr > 5 else "#16a34a"),
            _cell(f"{rmssd:.2f}", bold=best, color="#dc2626" if rmssd > 30 else "#16a34a"),
            _cell(f"{f1:.3f}",    bold=best, color="#1e293b"),
        ]))

    # Bar chart: HR MAE per combination
    labels   = [f"{pk[:3]}+{cl[:3]}" for pk, cl, *_ in _BIDMC_RESULTS]
    hr_maes  = [r[2] for r in _BIDMC_RESULTS]
    rmssd_es = [r[3] for r in _BIDMC_RESULTS]
    f1s      = [r[4] for r in _BIDMC_RESULTS]
    colors_hr   = ["#3b82f6" if (r[0], r[1]) == _BIDMC_BEST else "#93c5fd"
                   for r in _BIDMC_RESULTS]
    colors_rmssd = ["#3b82f6" if (r[0], r[1]) == _BIDMC_BEST else "#93c5fd"
                    for r in _BIDMC_RESULTS]

    fig_bar = go.Figure(data=[
        go.Bar(name="HR MAE (bpm)", x=labels, y=hr_maes,
               marker_color=colors_hr, text=[f"{v:.2f}" for v in hr_maes],
               textposition="outside", textfont=dict(size=10),
               hovertemplate="%{x}<br>HR MAE=%{y:.2f} bpm<extra></extra>"),
        go.Bar(name="RMSSD Δ (ms)", x=labels, y=[v/10 for v in rmssd_es],
               marker_color=["#f59e0b" if c == "#3b82f6" else "#fcd34d"
                              for c in colors_rmssd],
               text=[f"{v:.1f}" for v in rmssd_es],
               textposition="outside", textfont=dict(size=10),
               customdata=rmssd_es,
               hovertemplate="%{x}<br>RMSSD Δ=%{customdata:.2f} ms<extra></extra>"),
    ], layout=go.Layout(
        title=dict(text="HR MAE and RMSSD Δ by pipeline (lower is better)",
                   font=dict(size=12, color="#1e293b")),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(color="#334155", size=11),
        margin=dict(l=52, r=14, t=36, b=52),
        height=260,
        barmode="group",
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=True,
                   zerolinecolor="#e2e8f0", color="#334155",
                   title="bpm  (RMSSD axis ÷10 for scale)"),
        xaxis=dict(showgrid=False, color="#334155"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                    borderwidth=1, x=0.75, y=0.99, font=dict(size=10)),
        annotations=[dict(
            text="★ = best combination (Elgendi + Malik)",
            showarrow=False, x=0.01, y=1.06,
            xref="paper", yref="paper",
            font=dict(size=10, color="#64748b"), align="left")],
    ))

    fig_f1 = go.Figure(data=[
        go.Bar(name="F1", x=labels, y=f1s,
               marker_color=["#22c55e" if (r[0], r[1]) == _BIDMC_BEST else "#86efac"
                              for r in _BIDMC_RESULTS],
               text=[f"{v:.3f}" for v in f1s],
               textposition="outside", textfont=dict(size=10),
               hovertemplate="%{x}<br>F1=%{y:.3f}<extra></extra>"),
    ], layout=go.Layout(
        title=dict(text="Beat-level F1 (PAT-corrected, ±150 ms tolerance)",
                   font=dict(size=12, color="#1e293b")),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(color="#334155", size=11),
        margin=dict(l=52, r=14, t=36, b=52),
        height=260,
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                   color="#334155", title="F1", range=[0.92, 0.96]),
        xaxis=dict(showgrid=False, color="#334155"),
        showlegend=False,
    ))

    return html.Div(
        style=dict(padding="14px", display="flex", flexDirection="column",
                   gap="12px", overflowY="auto",
                   height="calc(100vh - 46px - 42px)"),
        children=[
            # Context card
            html.Div(style=_CARD, children=[
                html.Div("BIDMC Validation — Algorithm Selection Basis",
                         style=dict(fontSize="13px", fontWeight="700",
                                    color="#1e293b", marginBottom="8px")),
                html.Div(style=dict(display="flex", gap="24px", fontSize="12px",
                                    color="#475569", lineHeight="1.8"),
                children=[
                    html.Div([
                        html.Strong("Dataset: "), "BIDMC PPG & Respiration (PhysioNet, Pimentel 2017)", html.Br(),
                        html.Strong("Subjects: "), "53 ICU patients, 8 min each", html.Br(),
                        html.Strong("Signal: "), "PPG 125 Hz + synchronous ECG ground truth", html.Br(),
                        html.Strong("Ground truth: "), "ECG R-peaks (Hamilton 2002 detector, F1 > 0.99 on MIT-BIH)",
                    ], style=dict(flex="1")),
                    html.Div([
                        html.Strong("Fixed preprocessing: "),
                        "Butterworth 0.4–10 Hz, order 4, zero-phase", html.Br(),
                        html.Strong("Grid: "),
                        "2 peak detectors × 3 RR cleaners = 6 combinations", html.Br(),
                        html.Strong("Metrics: "),
                        "HR MAE / RMSSD Δ / F1 — reported together (Rule 4)", html.Br(),
                        html.Strong("Note: "),
                        "Bishop 2018 excluded — O(N²) complexity, ~10 h/record at N=60k",
                    ], style=dict(flex="1")),
                ]),
            ]),
            # Charts row
            html.Div(style=dict(display="flex", gap="10px"), children=[
                html.Div(style=dict(flex="3"),
                         children=dcc.Graph(figure=fig_bar,
                                            config=dict(displayModeBar=False),
                                            style=dict(height="260px"))),
                html.Div(style=dict(flex="2"),
                         children=dcc.Graph(figure=fig_f1,
                                            config=dict(displayModeBar=False),
                                            style=dict(height="260px"))),
            ]),
            # Ranking table
            html.Div(style=_CARD, children=[
                html.Div("Full Ranking — Median across 53 records",
                         style=dict(fontSize="12px", fontWeight="700",
                                    color="#1e293b", textTransform="uppercase",
                                    letterSpacing="0.05em", marginBottom="10px")),
                html.Table(
                    style=dict(borderCollapse="collapse", width="100%"),
                    children=[
                        html.Thead(html.Tr([
                            html.Th(h, style=dict(
                                padding="7px 14px",
                                textAlign="left" if h == "Pipeline" else "right",
                                color="#64748b", fontSize="11px",
                                textTransform="uppercase", letterSpacing="0.03em",
                                borderBottom="2px solid #e2e8f0"))
                            for h in cols
                        ])),
                        html.Tbody(table_rows),
                    ]),
                html.Div(
                    "★ Best: Elgendi + Malik  ·  "
                    "Criterion: HR MAE (primary) then RMSSD Δ (secondary). "
                    "F1 similar across all combinations (peak detection dominates; "
                    "RR cleaning does not affect beat-level F1, only IBI accuracy).",
                    style=dict(fontSize="11px", color="#94a3b8",
                               marginTop="10px", lineHeight="1.7")),
            ]),
        ],
    )


# ── Layout ─────────────────────────────────────────────────────────────────────

app.layout = html.Div(
    style=dict(backgroundColor="#f1f5f9", minHeight="100vh",
               fontFamily="'Inter','Segoe UI',system-ui,sans-serif"),
    children=[
        # ── Header ───────────────────────────────────────────────────────────
        html.Div(
            style=dict(backgroundColor="#0f172a", color="white",
                       padding="11px 24px", display="flex",
                       alignItems="center", gap="14px"),
            children=[
                html.Span("PPG", style=dict(color="#38bdf8", fontWeight="700",
                                             fontSize="17px")),
                html.Span("Algorithm Comparison Dashboard",
                          style=dict(fontWeight="500", fontSize="15px")),
                html.Span("P4-5 · population distributions",
                          style=dict(color="#64748b", fontSize="12px",
                                     marginLeft="auto")),
            ],
        ),
        # ── Body ─────────────────────────────────────────────────────────────
        html.Div(
            style=dict(display="flex", height="calc(100vh - 46px)"),
            children=[
                # ── Sidebar ───────────────────────────────────────────────────
                html.Div(
                    style=dict(width="260px", minWidth="260px",
                               backgroundColor="white",
                               borderRight="1px solid #e2e8f0",
                               padding="16px 14px",
                               display="flex", flexDirection="column", gap="14px",
                               overflowY="auto"),
                    children=[
                        html.Div([
                            _lbl("Subject"),
                            dcc.Dropdown(id="subject-dd", options=SUBJECTS,
                                         value=SUBJECTS[0]["value"] if SUBJECTS else None,
                                         clearable=False, style=_DD),
                        ]),
                        html.Div([
                            _lbl("Session"),
                            dcc.Dropdown(id="session-dd", options=[], value=None,
                                         clearable=False, style=_DD),
                        ]),
                        _sec("Filter Preset",
                            dcc.Dropdown(id="filter-dd",
                                options=[{"label": v[0], "value": k}
                                         for k, v in FILTER_PRESETS.items()],
                                value=FILTER_DEFAULT, clearable=False, style=_DD),
                        ),
                        html.Div([
                            _lbl("Peak Detection"),
                            dcc.Dropdown(id="peak-dd", options=PEAK_OPTIONS,
                                         value=PEAK_DEFAULT, clearable=False, style=_DD),
                        ]),
                        html.Div([
                            _lbl("RR Cleaning"),
                            dcc.Dropdown(id="clean-dd", options=CLEAN_OPTIONS,
                                         value=CLEAN_DEFAULT, clearable=False, style=_DD),
                        ]),
                        _sec("HRV Spectral Method",
                            dcc.Dropdown(id="psd-dd", options=PSD_OPTIONS,
                                         value=PSD_DEFAULT, clearable=False, style=_DD),
                        ),
                        html.Div(
                            id="session-info",
                            style=dict(backgroundColor="#f8fafc",
                                       border="1px solid #e2e8f0",
                                       borderRadius="8px", padding="10px",
                                       fontSize="12px", color="#475569",
                                       lineHeight="1.75", marginTop="4px"),
                            children="Select a session to begin.",
                        ),
                    ],
                ),
                # ── Main (tabbed) ─────────────────────────────────────────────
                html.Div(
                    style=dict(flex="1", display="flex", flexDirection="column",
                               overflow="hidden"),
                    children=[
                        dcc.Tabs(
                            id="main-tabs", value="session",
                            style=dict(backgroundColor="white",
                                       borderBottom="1px solid #e2e8f0"),
                            children=[
                                # ── Tab 3: Algorithm Validation (P4-6) ────
                                dcc.Tab(
                                    label="Algorithm Validation (BIDMC)",
                                    value="validation",
                                    style=dict(padding="8px 20px", fontSize="13px"),
                                    selected_style=dict(padding="8px 20px",
                                                        fontSize="13px",
                                                        fontWeight="600",
                                                        borderTop="2px solid #3b82f6"),
                                    children=_build_validation_tab(),
                                ),
                                # ── Tab 1: Session Analysis ────────────────
                                dcc.Tab(
                                    label="Session Analysis",
                                    value="session",
                                    style=dict(padding="8px 20px", fontSize="13px"),
                                    selected_style=dict(padding="8px 20px",
                                                        fontSize="13px",
                                                        fontWeight="600",
                                                        borderTop="2px solid #3b82f6"),
                                    children=html.Div(
                                        style=dict(padding="12px 14px",
                                                   display="flex", flexDirection="column",
                                                   gap="10px", overflowY="auto",
                                                   height="calc(100vh - 46px - 42px)"),
                                        children=[
                                            # HRV quick metrics
                                            html.Div(id="metrics-card", style=_CARD,
                                                     children=[html.Div(
                                                         "Select a session.",
                                                         style=dict(color="#94a3b8",
                                                                    fontSize="12px"))]),
                                            # Raw PPG
                                            dcc.Loading(type="circle", children=html.Div(
                                                style=_CARD,
                                                children=dcc.Graph(
                                                    id="raw-graph",
                                                    config=dict(displayModeBar=True,
                                                                scrollZoom=True),
                                                    style=dict(height="220px")))),
                                            # Filtered + peaks
                                            dcc.Loading(type="circle", children=html.Div(
                                                style=_CARD,
                                                children=dcc.Graph(
                                                    id="filtered-graph",
                                                    config=dict(displayModeBar=True,
                                                                scrollZoom=True),
                                                    style=dict(height="220px")))),
                                            # Window slider
                                            html.Div(
                                                style=dict(**_CARD, paddingBottom="18px"),
                                                children=[
                                                    _lbl("Display Window Start (s)"),
                                                    dcc.Slider(
                                                        id="window-slider",
                                                        min=0, max=100, step=1, value=0,
                                                        marks={},
                                                        tooltip={"placement": "bottom",
                                                                 "always_visible": True}),
                                                ]),
                                            # RR chart
                                            dcc.Loading(type="circle", children=html.Div(
                                                style=_CARD,
                                                children=dcc.Graph(
                                                    id="rr-graph",
                                                    config=dict(displayModeBar=True,
                                                                scrollZoom=True),
                                                    style=dict(height="180px")))),
                                            # HRV analysis panel (P4-3)
                                            html.Div(style=_CARD, children=[
                                                html.Div(
                                                    "HRV Analysis",
                                                    style=dict(fontSize="12px",
                                                               fontWeight="700",
                                                               color="#1e293b",
                                                               textTransform="uppercase",
                                                               letterSpacing="0.05em",
                                                               marginBottom="12px")),
                                                dcc.Loading(type="circle", children=
                                                    html.Div(
                                                        style=dict(display="flex", gap="10px"),
                                                        children=[
                                                            html.Div(style=dict(flex="3"),
                                                                children=dcc.Graph(
                                                                    id="psd-graph",
                                                                    config=dict(displayModeBar=True),
                                                                    style=dict(height="250px"))),
                                                            html.Div(style=dict(flex="2"),
                                                                children=dcc.Graph(
                                                                    id="poincare-graph",
                                                                    config=dict(displayModeBar=True),
                                                                    style=dict(height="250px"))),
                                                        ])),
                                                html.Div(id="hrv-table",
                                                         style=dict(marginTop="12px"),
                                                         children=html.Div(
                                                             "Select a session.",
                                                             style=dict(color="#94a3b8",
                                                                        fontSize="12px"))),
                                            ]),
                                        ],
                                    ),
                                ),
                                # ── Tab 2: Cross-Subject ───────────────────
                                dcc.Tab(
                                    label=f"Cross-Record ({N_BATCH})",
                                    value="cross",
                                    style=dict(padding="8px 20px", fontSize="13px"),
                                    selected_style=dict(padding="8px 20px",
                                                        fontSize="13px",
                                                        fontWeight="600",
                                                        borderTop="2px solid #3b82f6"),
                                    children=html.Div(
                                        style=dict(padding="12px 14px",
                                                   display="flex", flexDirection="column",
                                                   gap="10px", overflowY="auto",
                                                   height="calc(100vh - 46px - 42px)"),
                                        children=[
                                            # Batch controls
                                            html.Div(
                                                style=_CARD,
                                                children=[
                                                    html.Div(
                                                        style=dict(display="flex",
                                                                   alignItems="center",
                                                                   gap="16px"),
                                                        children=[
                                                            html.Button(
                                                                f"Compute All {N_BATCH} Records",
                                                                id="batch-btn",
                                                                n_clicks=0,
                                                                style=dict(
                                                                    backgroundColor="#3b82f6",
                                                                    color="white",
                                                                    border="none",
                                                                    borderRadius="6px",
                                                                    padding="8px 18px",
                                                                    fontSize="13px",
                                                                    fontWeight="600",
                                                                    cursor="pointer")),
                                                            html.Span(
                                                                id="batch-status",
                                                                style=dict(fontSize="12px",
                                                                           color="#64748b")),
                                                        ]),
                                                    html.Div(
                                                        "Uses the pipeline selected in the sidebar "
                                                        "(filter / peak / RR cleaning / spectral method). "
                                                        "First session per record; synthetic presets excluded.",
                                                        style=dict(fontSize="11px",
                                                                   color="#94a3b8",
                                                                   marginTop="6px")),
                                                ]),
                                            # Summary + scatter
                                            dcc.Loading(type="circle", children=
                                                html.Div(
                                                    style=dict(display="flex", gap="10px"),
                                                    children=[
                                                        html.Div(
                                                            id="cross-summary",
                                                            style=dict(**_CARD, flex="1"),
                                                            children=html.Div(
                                                                "Run batch compute to see summary.",
                                                                style=dict(color="#94a3b8",
                                                                           fontSize="12px"))),
                                                        html.Div(
                                                            style=dict(flex="3"),
                                                            children=dcc.Graph(
                                                                id="cross-scatter",
                                                                config=dict(displayModeBar=True),
                                                                style=dict(height="260px"))),
                                                    ])),
                                            # DataTable
                                            dcc.Loading(type="circle", children=
                                                html.Div(style=_CARD, children=[
                                                    html.Div(
                                                        "All Subjects — HRV Summary",
                                                        style=dict(fontSize="12px",
                                                                   fontWeight="700",
                                                                   color="#1e293b",
                                                                   textTransform="uppercase",
                                                                   letterSpacing="0.05em",
                                                                   marginBottom="10px")),
                                                    dash_table.DataTable(
                                                        id="cross-table",
                                                        columns=_TABLE_COLS,
                                                        data=[],
                                                        sort_action="native",
                                                        filter_action="native",
                                                        page_size=20,
                                                        style_table=dict(overflowX="auto"),
                                                        style_header=dict(
                                                            backgroundColor="#f8fafc",
                                                            fontWeight="600",
                                                            fontSize="11px",
                                                            color="#64748b",
                                                            border="1px solid #e2e8f0",
                                                            textTransform="uppercase",
                                                            letterSpacing="0.03em"),
                                                        style_cell=dict(
                                                            fontSize="12px",
                                                            color="#334155",
                                                            border="1px solid #f1f5f9",
                                                            padding="6px 10px",
                                                            textAlign="right"),
                                                        style_cell_conditional=[
                                                            {"if": {"column_id": "session"},
                                                             "textAlign": "left",
                                                             "maxWidth": "200px",
                                                             "overflow": "hidden",
                                                             "textOverflow": "ellipsis"},
                                                            {"if": {"column_id": "status"},
                                                             "textAlign": "left"},
                                                        ],
                                                        style_data_conditional=[
                                                            {"if": {"filter_query": '{status} = "OK"',
                                                                    "column_id": "status"},
                                                             "color": "#16a34a"},
                                                            {"if": {"filter_query": '{status} != "OK"',
                                                                    "column_id": "status"},
                                                             "color": "#dc2626"},
                                                            {"if": {"filter_query": "{removal_pct} > 20",
                                                                    "column_id": "removal_pct"},
                                                             "color": "#dc2626",
                                                             "fontWeight": "600"},
                                                        ],
                                                    ),
                                                ])),
                                            # Distribution plots (P4-5)
                                            dcc.Loading(type="circle", children=
                                                html.Div(style=_CARD, children=[
                                                    html.Div(
                                                        style=dict(
                                                            display="flex",
                                                            alignItems="center",
                                                            justifyContent="space-between",
                                                            marginBottom="10px"),
                                                        children=[
                                                            html.Div(
                                                                "Population Distributions",
                                                                style=dict(
                                                                    fontSize="12px",
                                                                    fontWeight="700",
                                                                    color="#1e293b",
                                                                    textTransform="uppercase",
                                                                    letterSpacing="0.05em")),
                                                            html.Span(
                                                                "Blue = Removal ≤ 20%  ·  "
                                                                "Orange = Removal > 20%",
                                                                style=dict(
                                                                    fontSize="11px",
                                                                    color="#94a3b8")),
                                                        ]),
                                                    dcc.Graph(
                                                        id="cross-dist",
                                                        config=dict(displayModeBar=True),
                                                        style=dict(height="320px")),
                                                ])),
                                        ],
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Store(id="session-store",       storage_type="memory"),
        dcc.Store(id="cross-subject-store", storage_type="memory"),
    ],
)


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("session-dd", "options"),
    Output("session-dd", "value"),
    Input("subject-dd", "value"),
)
def _cb_session_list(subject_id):
    if subject_id is None:
        return [], None
    opts = _list_sessions(str(subject_id))
    return opts, (opts[0]["value"] if opts else None)


@app.callback(
    Output("session-store", "data"),
    Output("window-slider", "max"),
    Output("window-slider", "value"),
    Output("window-slider", "step"),
    Output("session-info", "children"),
    Input("session-dd", "value"),
    Input("filter-dd", "value"),
    Input("peak-dd", "value"),
)
def _cb_load_session(source_key, filter_key, peak_method):
    empty = "Select a session to begin."
    if not source_key or not filter_key or not peak_method:
        return None, 100, 0, 5, empty
    try:
        data = _load_and_cache(source_key, filter_key, peak_method)
    except Exception as exc:
        return None, 100, 0, 5, f"Error: {exc}"
    key       = f"{source_key}::{filter_key}::{peak_method}"
    max_start = max(0.0, data["duration_s"] - WINDOW_S)
    step      = max(1.0, round(max_start / 100, 1))
    n_peaks   = len(data["peaks"])
    hr_est    = n_peaks / data["duration_s"] * 60 if data["duration_s"] > 0 else 0
    flabel    = FILTER_PRESETS[filter_key][0].split("  ")[0]
    info = [
        html.Strong(_session_label(source_key)), html.Br(),
        f"Duration: {data['duration_s']:.1f} s  ·  {data['n_samples']:,} samples", html.Br(),
        f"fs: {data['fs']} Hz  ·  {flabel}", html.Br(),
        f"Peaks: {n_peaks}  ·  HR est: {hr_est:.0f} bpm",
    ]
    return {"key": key}, max_start, 0.0, step, info


@app.callback(
    Output("raw-graph", "figure"),
    Output("filtered-graph", "figure"),
    Input("session-store", "data"),
    Input("window-slider", "value"),
)
def _cb_signal_graphs(store, window_start):
    _base = dict(paper_bgcolor="white", plot_bgcolor="#f8fafc",
                 font=dict(color="#334155", size=11),
                 margin=dict(l=52, r=14, t=32, b=32),
                 xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                            color="#334155", title="Time (s)"),
                 yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                            color="#334155", title="mV"),
                 height=208, hovermode="x unified")

    def _empty(title):
        return go.Figure(layout=go.Layout(
            title=dict(text=title, font=dict(size=12, color="#1e293b")),
            annotations=[dict(text="No data", showarrow=False,
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              font=dict(size=13, color="#94a3b8"))],
            **_base))

    if not store or "key" not in store:
        return _empty("Raw PPG"), _empty("Filtered PPG + Peaks")
    data = _SESSION_CACHE.get(store["key"])
    if data is None:
        return _empty("Raw PPG"), _empty("Filtered PPG + Peaks")

    unit = data.get("unit", "mV")
    _base["yaxis"] = {**_base["yaxis"], "title": unit}
    t0   = float(window_start or 0.0)
    t1   = t0 + WINDOW_S
    ts   = data["time_s"]
    mask = (ts >= t0) & (ts <= t1)
    t_w  = ts[mask]
    lo   = int(np.searchsorted(ts, t0))
    hi   = int(np.searchsorted(ts, t1, side="right"))
    pk   = data["peaks"][(data["peaks"] >= lo) & (data["peaks"] < hi)]
    xax  = dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                color="#334155", title="Time (s)", range=[t0, t1])

    fig_raw = go.Figure(
        data=[go.Scatter(x=t_w, y=data["ppg_raw"][mask], mode="lines",
                         line=dict(color="#0ea5e9", width=1.2), name="Raw PPG",
                         hovertemplate="%{y:.3f} " + unit + "<extra></extra>")],
        layout=go.Layout(title=dict(text="Raw PPG", font=dict(size=12, color="#1e293b")),
                         showlegend=False, xaxis=xax,
                         **{k: v for k, v in _base.items() if k != "xaxis"}))

    ftraces = [go.Scatter(x=t_w, y=data["ppg_filt"][mask], mode="lines",
                           line=dict(color="#22c55e", width=1.2), name="Filtered",
                           hovertemplate="%{y:.3f} " + unit + "<extra></extra>")]
    if len(pk) > 0:
        ftraces.append(go.Scatter(
            x=ts[pk], y=data["ppg_filt"][pk], mode="markers",
            marker=dict(color="#ef4444", size=7, symbol="circle-open",
                        line=dict(width=2)),
            name="Peaks", hovertemplate="t=%{x:.3f} s<extra>Peak</extra>"))

    fig_filt = go.Figure(
        data=ftraces,
        layout=go.Layout(
            title=dict(text="Filtered PPG + Peaks", font=dict(size=12, color="#1e293b")),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                        borderwidth=1, x=0.01, y=0.99, font=dict(size=11)),
            xaxis=xax,
            **{k: v for k, v in _base.items() if k != "xaxis"}))
    return fig_raw, fig_filt


@app.callback(
    Output("rr-graph", "figure"),
    Output("metrics-card", "children"),
    Input("session-store", "data"),
    Input("clean-dd", "value"),
)
def _cb_rr_metrics(store, clean_method):
    _el = go.Figure(layout=go.Layout(
        title=dict(text="RR Intervals", font=dict(size=12, color="#1e293b")),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(color="#334155", size=11),
        margin=dict(l=52, r=14, t=32, b=32),
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                   color="#334155", title="Time (s)"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                   color="#334155", title="RR (ms)"),
        height=168))
    _ec = html.Div("Select a session.", style=dict(color="#94a3b8", fontSize="12px"))

    if not store or "key" not in store or not clean_method:
        return _el, _ec
    data = _SESSION_CACHE.get(store["key"])
    if data is None:
        return _el, _ec

    m = _compute_rr_metrics(data, clean_method)
    if "error" in m:
        return _el, html.Div(f"Error: {m['error']}",
                              style=dict(color="#ef4444", fontSize="12px"))

    rr_t  = m["peak_times_s"][1:]
    mask  = m["rr_mask"]
    phy_lo = float(CFG["rr_cleaning"]["physiological_min_ms"])
    phy_hi = float(CFG["rr_cleaning"]["physiological_max_ms"])
    span   = [float(rr_t[0]), float(rr_t[-1])] if len(rr_t) > 0 else [0, 1]

    rr_traces = [
        go.Scatter(x=rr_t[mask], y=m["rr_raw_ms"][mask], mode="markers",
                   marker=dict(color="#22c55e", size=4, opacity=0.75),
                   name=f"Kept ({mask.sum()})",
                   hovertemplate="t=%{x:.2f}s  RR=%{y:.0f}ms<extra>Kept</extra>"),
    ]
    if (~mask).sum() > 0:
        rr_traces.append(go.Scatter(
            x=rr_t[~mask], y=m["rr_raw_ms"][~mask], mode="markers",
            marker=dict(color="#ef4444", size=6, symbol="x", opacity=0.9),
            name=f"Removed ({(~mask).sum()})",
            hovertemplate="t=%{x:.2f}s  RR=%{y:.0f}ms<extra>Removed</extra>"))
    rr_traces.append(go.Scatter(
        x=[span[0], span[1], span[1], span[0], span[0]],
        y=[phy_lo,  phy_lo,  phy_hi,  phy_hi,  phy_lo],
        mode="lines", fill="toself",
        fillcolor="rgba(148,163,184,0.07)",
        line=dict(color="rgba(148,163,184,0.25)", width=1, dash="dot"),
        name="Phys. range", hoverinfo="skip"))

    fig_rr = go.Figure(
        data=rr_traces,
        layout=go.Layout(
            title=dict(text="RR Intervals — full session",
                       font=dict(size=12, color="#1e293b")),
            paper_bgcolor="white", plot_bgcolor="#f8fafc",
            font=dict(color="#334155", size=11),
            margin=dict(l=52, r=14, t=32, b=32),
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                       color="#334155", title="Time (s)"),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                       color="#334155", title="RR (ms)"),
            height=168,
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                        borderwidth=1, x=0.01, y=0.99, font=dict(size=10),
                        orientation="h")))

    def _f(v, d=1):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f"{v:.{d}f}"

    parts = store["key"].split("::")
    flabel = FILTER_PRESETS.get(parts[1] if len(parts)>1 else "", ("?",))[0].split("  ")[0]
    pm     = parts[2] if len(parts) > 2 else "?"
    tag    = f"{flabel}  +  {pm}  +  {clean_method}"
    rc     = "#ef4444" if m["removal_pct"] > 20 else "#475569"

    card = [
        html.Div(style=dict(display="flex", alignItems="center",
                            justifyContent="space-between", marginBottom="10px"),
                 children=[
                     html.Span("HRV Metrics", style=dict(
                         fontSize="12px", fontWeight="700", color="#1e293b",
                         textTransform="uppercase", letterSpacing="0.05em")),
                     html.Span(tag, style=dict(
                         fontSize="11px", color="#64748b",
                         backgroundColor="#f1f5f9", borderRadius="4px",
                         padding="2px 8px")),
                 ]),
        html.Div(style=dict(display="flex", gap="8px", flexWrap="wrap"),
                 children=[
                     _metric(_f(m.get("mean_hr_bpm")), "bpm",    "HR mean"),
                     _metric(_f(m.get("rmssd_ms")),    "ms",     "RMSSD"),
                     _metric(_f(m.get("sdnn_ms")),     "ms",     "SDNN"),
                     _metric(_f(m.get("pnn50_pct")),   "%",      "pNN50"),
                     _metric(str(m["n_peaks"]),         "peaks",  "Detected",   "#0ea5e9"),
                     _metric(str(m["n_rr_clean"]),      "RR",     "After clean","#22c55e"),
                     _metric(f"{m['n_rr_removed']} ({m['removal_pct']:.0f}%)",
                             "removed", "Cleaned out", rc),
                 ]),
    ]
    return fig_rr, card


@app.callback(
    Output("psd-graph", "figure"),
    Output("poincare-graph", "figure"),
    Output("hrv-table", "children"),
    Input("session-store", "data"),
    Input("clean-dd", "value"),
    Input("psd-dd", "value"),
)
def _cb_hrv_panel(store, clean_method, psd_method):
    _bl = dict(paper_bgcolor="white", plot_bgcolor="#f8fafc",
               font=dict(color="#334155", size=11),
               margin=dict(l=52, r=14, t=36, b=36), height=238)

    def _ef(title):
        return go.Figure(layout=go.Layout(
            title=dict(text=title, font=dict(size=12, color="#1e293b")),
            annotations=[dict(text="No data", showarrow=False,
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              font=dict(size=13, color="#94a3b8"))],
            **_bl))

    empty_tbl = html.Div("Select a session.", style=dict(color="#94a3b8",
                                                          fontSize="12px"))
    if not store or "key" not in store or not clean_method or not psd_method:
        return _ef("PSD"), _ef("Poincaré"), empty_tbl

    data = _SESSION_CACHE.get(store["key"])
    if data is None:
        return _ef("PSD"), _ef("Poincaré"), empty_tbl

    m = _compute_rr_metrics(data, clean_method)
    if "error" in m or len(m["rr_clean_ms"]) < 4:
        msg = m.get("error", "Too few RR")
        return (_ef(f"PSD — {msg[:40]}"), _ef("Poincaré"),
                html.Div(f"Error: {msg}", style=dict(color="#ef4444", fontSize="12px")))

    rr_clean = m["rr_clean_ms"]

    # ── PSD ──────────────────────────────────────────────────────────────────
    fr = _compute_hrv_freq_metrics(rr_clean, psd_method)
    if "error" in fr:
        fig_psd = _ef(f"PSD — {fr['error'][:40]}")
    else:
        freqs, psd_v = fr["freqs"], fr["psd"]
        bp = fr["band_powers"]

        def _shade(lo, hi, color, label):
            mk = (freqs >= lo) & (freqs <= hi)
            if not mk.any():
                return None
            x = np.concatenate([[lo], freqs[mk], [hi]])
            y = np.concatenate([[0],  psd_v[mk], [0]])
            return go.Scatter(x=x, y=y, mode="lines", fill="toself",
                               fillcolor=color, line=dict(width=0),
                               name=label, hoverinfo="skip")

        vlo, vhi = HRV_BANDS["vlf"]
        llo, lhi = HRV_BANDS["lf"]
        hlo, hhi = HRV_BANDS["hf"]

        psd_traces = [s for s in [
            _shade(vlo, vhi, "rgba(167,139,250,0.18)", "VLF"),
            _shade(llo, lhi, "rgba(56,189,248,0.20)",  "LF"),
            _shade(hlo, hhi, "rgba(34,197,94,0.18)",   "HF"),
        ] if s is not None]
        psd_traces.append(go.Scatter(
            x=freqs, y=psd_v, mode="lines",
            line=dict(color="#3b82f6", width=1.5), name="PSD", showlegend=False,
            hovertemplate="f=%{x:.3f} Hz  PSD=%{y:.1f}<extra></extra>"))

        def _bstr(k):
            v = bp.get(f"{k}_ms2")
            return f"{v:.1f}" if v and np.isfinite(v) else "—"

        lf_hf = bp.get("lf_hf_ratio")
        lf_hf_s = f"{lf_hf:.2f}" if lf_hf and np.isfinite(lf_hf) else "—"

        fig_psd = go.Figure(
            data=psd_traces,
            layout=go.Layout(
                title=dict(text=f"PSD — {psd_method}",
                           font=dict(size=12, color="#1e293b")),
                annotations=[dict(
                    text=(f"VLF: {_bstr('vlf')} ms²  "
                          f"LF: {_bstr('lf')} ms²  "
                          f"HF: {_bstr('hf')} ms²  "
                          f"LF/HF: {lf_hf_s}"),
                    showarrow=False, x=0.5, y=1.04,
                    xref="paper", yref="paper",
                    font=dict(size=10, color="#64748b"), align="center")],
                xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                           color="#334155", title="Frequency (Hz)", range=[0, 0.5]),
                yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=True,
                           zerolinecolor="#e2e8f0", color="#334155", title="ms²/Hz"),
                legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                            borderwidth=1, x=0.6, y=0.99, font=dict(size=10)),
                **_bl))

    # ── Poincaré ──────────────────────────────────────────────────────────────
    nl = {}
    try:
        nl = compute_hrv_nonlinear(rr_clean)
        rr_n, rr_n1 = nl["rr_n"], nl["rr_n1"]
        sd1, sd2 = nl["sd1_ms"], nl["sd2_ms"]
        mrr = float(np.mean(rr_clean))

        th  = np.linspace(0, 2 * np.pi, 200)
        c45, s45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
        ex = mrr + sd2 * np.cos(th) * c45 - sd1 * np.sin(th) * s45
        ey = mrr + sd2 * np.cos(th) * s45 + sd1 * np.sin(th) * c45
        plim = max(sd2 * 1.2, 100.0)

        fig_pc = go.Figure(
            data=[
                go.Scatter(x=[mrr - plim*c45, mrr + plim*c45],
                           y=[mrr - plim*s45, mrr + plim*s45],
                           mode="lines",
                           line=dict(color="#cbd5e1", width=1, dash="dot"),
                           showlegend=False, hoverinfo="skip"),
                go.Scatter(x=[mrr - plim*c45, mrr + plim*c45],
                           y=[mrr + plim*s45, mrr - plim*s45],
                           mode="lines",
                           line=dict(color="#cbd5e1", width=1, dash="dot"),
                           showlegend=False, hoverinfo="skip"),
                go.Scatter(x=rr_n, y=rr_n1, mode="markers",
                           marker=dict(color="#3b82f6", size=4, opacity=0.55),
                           name=f"RR pairs (n={len(rr_n)})",
                           hovertemplate="RRn=%{x:.0f}  RRn+1=%{y:.0f}<extra></extra>"),
                go.Scatter(x=ex, y=ey, mode="lines",
                           line=dict(color="#f59e0b", width=2),
                           name=f"SD1={sd1:.1f}  SD2={sd2:.1f} ms"),
            ],
            layout=go.Layout(
                title=dict(text="Poincaré Plot", font=dict(size=12, color="#1e293b")),
                xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                           color="#334155", title="RRn (ms)",
                           range=[float(np.min(rr_n))-50, float(np.max(rr_n))+50],
                           scaleanchor="y"),
                yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                           color="#334155", title="RRn+1 (ms)",
                           range=[float(np.min(rr_n))-50, float(np.max(rr_n))+50]),
                legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                            borderwidth=1, x=0.01, y=0.99, font=dict(size=9)),
                **_bl))
    except Exception as exc:
        fig_pc = _ef(f"Poincaré — {str(exc)[:40]}")

    # ── Time-domain table ─────────────────────────────────────────────────────
    def _f(v, d=1):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f"{v:.{d}f}"

    def _tbl(rows):
        return html.Table(
            style=dict(borderCollapse="collapse", width="100%"),
            children=[
                html.Thead(html.Tr([
                    html.Th(h, style=dict(padding="5px 10px", textAlign=a,
                                          color="#64748b", fontSize="11px",
                                          borderBottom="2px solid #e2e8f0"))
                    for h, a in [("Metric", "left"), ("Value", "right"), ("Unit", "left")]
                ])),
                html.Tbody(rows),
            ])

    rows_l = [
        _trow("MeanNN",  _f(m.get("mean_nn_ms")), "ms",  hi=True),
        _trow("SDNN",    _f(m.get("sdnn_ms")),    "ms"),
        _trow("RMSSD",   _f(m.get("rmssd_ms")),   "ms",  hi=True),
        _trow("pNN50",   _f(m.get("pnn50_pct")),  "%"),
        _trow("NN50",    str(m.get("nn50", "—")),  "beats"),
    ]
    rows_r = [
        _trow("MeanHR",  _f(m.get("mean_hr_bpm")), "bpm", hi=True),
        _trow("MinHR",   _f(m.get("min_hr_bpm")),  "bpm"),
        _trow("MaxHR",   _f(m.get("max_hr_bpm")),  "bpm"),
        _trow("SD1",     _f(nl.get("sd1_ms")),      "ms",  hi=True),
        _trow("SD2",     _f(nl.get("sd2_ms")),      "ms"),
        _trow("SD1/SD2", _f(nl.get("sd1_sd2"), 3),  ""),
    ]

    hrv_table = html.Div([
        html.Div("Time-Domain Metrics", style=dict(
            fontSize="11px", fontWeight="600", color="#64748b",
            textTransform="uppercase", letterSpacing="0.04em",
            marginBottom="8px")),
        html.Div(style=dict(display="flex", gap="16px"),
                 children=[
                     html.Div(style=dict(flex="1"), children=_tbl(rows_l)),
                     html.Div(style=dict(flex="1"), children=_tbl(rows_r)),
                 ]),
    ])
    return fig_psd, fig_pc, hrv_table


@app.callback(
    Output("cross-subject-store", "data"),
    Output("batch-status", "children"),
    Input("batch-btn", "n_clicks"),
    State("filter-dd",  "value"),
    State("peak-dd",    "value"),
    State("clean-dd",   "value"),
    State("psd-dd",     "value"),
    prevent_initial_call=True,
)
def _cb_batch_compute(n_clicks, filter_key, peak_method, clean_method, psd_method):
    import time
    t0 = time.time()
    results = _batch_compute(filter_key, peak_method, clean_method, psd_method)
    elapsed = time.time() - t0
    ok_count = sum(1 for r in results if r.get("status") == "OK")
    flabel = FILTER_PRESETS.get(filter_key, ("?",))[0].split("  ")[0]
    status = (f"Computed {ok_count}/{len(results)} records in {elapsed:.1f}s  ·  "
              f"Pipeline: {flabel} + {peak_method} + {clean_method} + {psd_method}")
    return results, status


@app.callback(
    Output("cross-table", "data"),
    Output("cross-scatter", "figure"),
    Output("cross-summary", "children"),
    Input("cross-subject-store", "data"),
)
def _cb_cross_view(results):
    empty_scatter = go.Figure(layout=go.Layout(
        title=dict(text="RMSSD vs HR — run batch compute first",
                   font=dict(size=12, color="#1e293b")),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(color="#334155", size=11),
        margin=dict(l=52, r=14, t=36, b=36), height=248,
        xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                   color="#334155", title="HR (bpm)"),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                   color="#334155", title="RMSSD (ms)")))
    empty_summary = html.Div("Run batch compute to see summary.",
                             style=dict(color="#94a3b8", fontSize="12px"))

    if not results:
        return [], empty_scatter, empty_summary

    # ── Scatter ───────────────────────────────────────────────────────────────
    ok = [r for r in results if r.get("status") == "OK"
          and r.get("hr_mean") is not None and r.get("rmssd_ms") is not None]

    scatter = go.Figure(
        data=[go.Scatter(
            x=[r["hr_mean"]  for r in ok],
            y=[r["rmssd_ms"] for r in ok],
            mode="markers+text" if len(ok) <= 20 else "markers",  # labels collide beyond ~20
            marker=dict(
                size=8, opacity=0.8,
                color=list(range(len(ok))),
                colorscale="Viridis",
                showscale=False,
            ),
            text=[r["short"] for r in ok],
            textposition="top center",
            textfont=dict(size=9, color="#64748b"),
            customdata=[[r["subject"], r["sdnn_ms"], r["pnn50_pct"],
                         r["removal_pct"], r.get("lf_hf")] for r in ok],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "HR: %{x:.1f} bpm<br>"
                "RMSSD: %{y:.1f} ms<br>"
                "SDNN: %{customdata[1]} ms<br>"
                "pNN50: %{customdata[2]}%<br>"
                "Removed: %{customdata[3]}%<br>"
                "LF/HF: %{customdata[4]}<extra></extra>"
            ),
        )],
        layout=go.Layout(
            title=dict(text="RMSSD vs HR — all records (first session each)",
                       font=dict(size=12, color="#1e293b")),
            paper_bgcolor="white", plot_bgcolor="#f8fafc",
            font=dict(color="#334155", size=11),
            margin=dict(l=52, r=80, t=36, b=36), height=248,
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                       color="#334155", title="HR mean (bpm)"),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                       color="#334155", title="RMSSD (ms)"),
        ),
    )

    # ── Summary card ──────────────────────────────────────────────────────────
    def _agg(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if not vals:
            return "—", "—", "—"
        a = np.array(vals, dtype=float)
        return f"{np.mean(a):.1f}", f"{np.min(a):.1f}", f"{np.max(a):.1f}"

    hr_m,  hr_lo,  hr_hi  = _agg("hr_mean")
    rms_m, rms_lo, rms_hi = _agg("rmssd_ms")
    sdn_m, sdn_lo, sdn_hi = _agg("sdnn_ms")
    rem_m, _,      _       = _agg("removal_pct")

    summary = html.Div([
        html.Div(f"{len(ok)} / {len(results)} records OK",
                 style=dict(fontSize="14px", fontWeight="700", color="#1e293b",
                            marginBottom="10px")),
        html.Table(
            style=dict(borderCollapse="collapse", width="100%", fontSize="12px"),
            children=[
                html.Thead(html.Tr([
                    html.Th(h, style=dict(padding="4px 8px", textAlign="right",
                                          color="#64748b", fontSize="11px",
                                          borderBottom="1px solid #e2e8f0"))
                    for h in ["Metric", "Mean", "Min", "Max"]
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(lbl, style=dict(padding="4px 8px", color="#475569")),
                        html.Td(mean, style=dict(padding="4px 8px", textAlign="right",
                                                  fontWeight="600")),
                        html.Td(lo,   style=dict(padding="4px 8px", textAlign="right",
                                                  color="#94a3b8")),
                        html.Td(hi,   style=dict(padding="4px 8px", textAlign="right",
                                                  color="#94a3b8")),
                    ])
                    for lbl, mean, lo, hi in [
                        ("HR (bpm)",   hr_m,  hr_lo,  hr_hi),
                        ("RMSSD (ms)", rms_m, rms_lo, rms_hi),
                        ("SDNN (ms)",  sdn_m, sdn_lo, sdn_hi),
                        ("Removed%",   rem_m, "—",    "—"),
                    ]
                ]),
            ]),
    ])

    return results, scatter, summary


@app.callback(
    Output("cross-dist", "figure"),
    Input("cross-subject-store", "data"),
)
def _cb_distribution(results):
    _METRICS = [
        ("hr_mean",   "HR (bpm)"),
        ("rmssd_ms",  "RMSSD (ms)"),
        ("sdnn_ms",   "SDNN (ms)"),
        ("pnn50_pct", "pNN50 (%)"),
    ]
    empty = go.Figure(layout=go.Layout(
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(color="#334155", size=11),
        margin=dict(l=40, r=14, t=36, b=36), height=308,
        annotations=[dict(text="Run batch compute to see distributions.",
                          showarrow=False, x=0.5, y=0.5,
                          xref="paper", yref="paper",
                          font=dict(size=13, color="#94a3b8"))]))
    if not results:
        return empty

    ok = [r for r in results if r.get("status") == "OK"]
    if not ok:
        return empty

    clean   = [r for r in ok if (r.get("removal_pct") or 0) <= 20]
    flagged = [r for r in ok if (r.get("removal_pct") or 0) >  20]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[m[1] for m in _METRICS],
        shared_yaxes=False,
    )

    for col_idx, (key, label) in enumerate(_METRICS, start=1):
        vals_c = [r[key] for r in clean   if r.get(key) is not None]
        vals_f = [r[key] for r in flagged if r.get(key) is not None]

        # Jitter x offset so points don't overlap boxes
        rng = np.random.default_rng(42)

        if vals_c:
            jitter_c = rng.uniform(-0.15, 0.15, len(vals_c))
            fig.add_trace(go.Box(
                y=vals_c, name="≤20% removed",
                marker_color="#3b82f6",
                line_color="#2563eb",
                fillcolor="rgba(59,130,246,0.15)",
                boxpoints=False,
                showlegend=(col_idx == 1),
                legendgroup="clean",
            ), row=1, col=col_idx)
            fig.add_trace(go.Scatter(
                x=[1 + j for j in jitter_c], y=vals_c,
                mode="markers",
                marker=dict(color="#3b82f6", size=5, opacity=0.55,
                            line=dict(width=0)),
                showlegend=False,
                hovertemplate=f"%{{y:.1f}} {label}<extra></extra>",
                xaxis=f"x{col_idx}", yaxis=f"y{col_idx}",
            ), row=1, col=col_idx)

        if vals_f:
            jitter_f = rng.uniform(-0.15, 0.15, len(vals_f))
            fig.add_trace(go.Box(
                y=vals_f, name=">20% removed",
                marker_color="#f97316",
                line_color="#ea580c",
                fillcolor="rgba(249,115,22,0.15)",
                boxpoints=False,
                showlegend=(col_idx == 1),
                legendgroup="flagged",
            ), row=1, col=col_idx)
            fig.add_trace(go.Scatter(
                x=[1 + j for j in jitter_f], y=vals_f,
                mode="markers",
                marker=dict(color="#f97316", size=5, opacity=0.65,
                            line=dict(width=0)),
                showlegend=False,
                hovertemplate=f"%{{y:.1f}} {label}<extra></extra>",
                xaxis=f"x{col_idx}", yaxis=f"y{col_idx}",
            ), row=1, col=col_idx)

        # Annotation: median (IQR)
        all_vals = vals_c + vals_f
        if all_vals:
            a = np.array(all_vals)
            med = float(np.median(a))
            q1  = float(np.percentile(a, 25))
            q3  = float(np.percentile(a, 75))
            fig.add_annotation(
                text=f"med {med:.1f}<br>[{q1:.1f}, {q3:.1f}]",
                xref=f"x{col_idx} domain", yref=f"y{col_idx} domain",
                x=0.5, y=-0.22, showarrow=False,
                font=dict(size=9, color="#64748b"), align="center",
                row=1, col=col_idx,
            )

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        font=dict(color="#334155", size=11),
        margin=dict(l=40, r=14, t=40, b=60),
        height=308,
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0",
                    borderwidth=1, x=1.01, y=1, font=dict(size=10),
                    orientation="v"),
        showlegend=True,
    )
    for i in range(1, 5):
        axis = f"yaxis{i}" if i > 1 else "yaxis"
        xaxis = f"xaxis{i}" if i > 1 else "xaxis"
        fig.update_layout(**{
            axis: dict(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                       color="#334155"),
            xaxis: dict(showticklabels=False, showgrid=False, zeroline=False),
        })
    fig.update_annotations(font_size=11)

    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
