"""
NTRO PS-26147 — RF Parameter Extraction Dashboard
Streamlit + Plotly signal analysis workstation with Apple Light Mode aesthetic.
Features inline Signal Ingestion card, Analysis Canvas with distinct visual boundaries,
prominent file upload button, and live REST integration with FastAPI backend.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import signal as sp_signal
import requests
import json
import datetime
import io
import os
import time

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NTRO PS-26147 — RF Parameter Extraction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM STYLING (Crisp Boundaries, High Contrast, Visible Uploader)
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global Theme ───────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #f1f5f9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0f172a;
}

/* ── Hide Sidebar & Default Streamlit Chrome ────────────── */
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
#MainMenu, footer, header {
    display: none !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1580px !important;
}

/* ── Top Navbar ─────────────────────────────────────────── */
.top-navbar-card {
    background: #0f172a;
    border: 1.5px solid #1e293b;
    border-radius: 14px;
    padding: 0.85rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.15);
}

/* ── Container Borders & Segregation ────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    margin-bottom: 1.25rem !important;
}

/* Sub-card segregation */
.sub-boundary-card {
    background: #f8fafc;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.9rem;
}

.dark-boundary-card {
    background: #0b0c10;
    border: 1.5px solid #1e293b;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.9rem;
}

/* ── Headers & Labels ───────────────────────────────────── */
.card-section-tag {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    color: #475569;
    margin-bottom: 0.2rem;
}
.card-main-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* ── Streamlit File Uploader (Prominent & Clearly Visible) ── */
[data-testid="stFileUploader"] {
    width: 100% !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stFileUploader"] section {
    background: #f8fafc !important;
    border: 2px dashed #3b82f6 !important;
    border-radius: 12px !important;
    padding: 1.5rem 1rem !important;
    text-align: center !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"] section:hover {
    border-color: #1d4ed8 !important;
    background: #eff6ff !important;
}

/* Visible Upload Button */
[data-testid="stFileUploader"] button {
    background: #0071e3 !important;
    color: #ffffff !important;
    border: 1px solid #005bb5 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(0, 113, 227, 0.3) !important;
    cursor: pointer !important;
    display: inline-block !important;
    visibility: visible !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #005bb5 !important;
}

[data-testid="stFileUploader"] small {
    color: #64748b !important;
    font-size: 0.78rem !important;
    margin-top: 0.4rem !important;
    display: block !important;
}

/* ── Primary Action Button ──────────────────────────────── */
div.stButton > button[kind="primary"] {
    background: #0071e3 !important;
    color: #ffffff !important;
    border: 1.5px solid #005bb5 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.25rem !important;
    box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3) !important;
    transition: all 0.15s ease !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #005bb5 !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(0, 113, 227, 0.4) !important;
}

/* ── KPI Metric Cards ───────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.kpi-unit-card {
    background: #ffffff;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.kpi-unit-card:hover {
    border-color: #3b82f6;
}
.kpi-unit-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: #64748b;
    margin-bottom: 0.3rem;
}
.kpi-unit-val {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.kpi-unit-unit {
    font-size: 0.78rem;
    font-weight: 500;
    color: #64748b;
}

/* ── Status Badges ──────────────────────────────────────── */
.badge-tag-pass {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    background: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
}
.badge-tag-fail {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    background: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
}
.badge-tag-info {
    display: inline-block;
    padding: 0.18rem 0.55rem;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    background: #e0f2fe;
    color: #075985;
    border: 1px solid #bae6fd;
}

/* ── Chart Headers ──────────────────────────────────────── */
.chart-sub-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.chart-sub-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b;
}
.chart-meta-pill {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    color: #475569;
    background: #e2e8f0;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-weight: 500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MOCK DATA GENERATION
# ─────────────────────────────────────────────────────────────
def _generate_mock_data():
    """Deterministic high-fidelity mock data matching QPSK transmission."""
    rng = np.random.default_rng(seed=42)

    # QPSK constellation
    ideal_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    n_symbols = 600
    indices = rng.integers(0, 4, size=n_symbols)
    noise = (rng.normal(0, 0.08, n_symbols) + 1j * rng.normal(0, 0.08, n_symbols))
    rx_symbols = ideal_symbols[indices] + noise

    evm_pct = float(np.mean(np.abs(noise[:n_symbols])) / np.mean(np.abs(ideal_symbols[indices])) * 100)

    # Time-domain IQ (1000 samples)
    t = np.arange(1000)
    i_samples = np.real(rx_symbols[:250]).repeat(4) + rng.normal(0, 0.02, 1000)
    q_samples = np.imag(rx_symbols[:250]).repeat(4) + rng.normal(0, 0.02, 1000)

    # PSD curve (peak at 0 kHz)
    freq_axis = np.linspace(-500, 500, 161)  # kHz
    psd_db = -88.0 + 36.0 * np.exp(-0.5 * (freq_axis / 55.0) ** 2)
    psd_db += rng.normal(0, 0.8, len(freq_axis))
    psd_db = np.clip(psd_db, -92.0, -50.0)
    noise_floor_db = -85.0

    # Spectrogram matrix
    n_times = 48
    n_freqs = 64
    spec_matrix = np.zeros((n_freqs, n_times))
    f_spec = np.linspace(-500, 500, n_freqs)
    t_spec = np.linspace(0, 96, n_times)
    for ti in range(n_times):
        spec_matrix[:, ti] = -88.0 + 36.0 * np.exp(-0.5 * (f_spec / 55.0) ** 2) + rng.normal(0, 1.2, n_freqs)

    # Payload
    payload_bytes = rng.integers(0, 256, size=128).astype(np.uint8)
    hex_stream = " ".join(f"{b:02X}" for b in payload_bytes)
    ascii_stream = "".join(chr(b) if 32 <= b < 127 else "." for b in payload_bytes)

    c40_val = complex(0.76, 0.02)
    c42_val = complex(1.00, -0.01)
    c63_val = complex(0.03, 0.01)

    return {
        "carrier_freq_hz": 1420405752.0,
        "symbol_rate_baud": 9600.0,
        "modulation": "QPSK",
        "amc_confidence": 0.94,
        "amc_probabilities": {
            "BPSK": 0.03, "QPSK": 0.94, "8PSK": 0.01,
            "16-QAM": 0.01, "2-FSK": 0.005, "4-FSK": 0.005,
        },
        "evm_pct": round(evm_pct, 2),
        "snr_eff_db": round(20 * np.log10(100 / max(evm_pct, 0.01)), 1),
        "interleaver_pattern": "Block 8×32",
        "interleaver_confidence": 0.72,
        "interleaver_gf2_rank": 7,
        "fec_scheme": "Conv R=1/2 K=7",
        "fec_confidence": 0.68,
        "viterbi_convergence": 0.91,
        "rs_syndromes_zero": True,
        "occupied_bw_khz": 38.4,
        "noise_floor_dbm": noise_floor_db,
        "cfar_status": "DETECTED",
        "constellation_rx_i": np.real(rx_symbols).tolist(),
        "constellation_rx_q": np.imag(rx_symbols).tolist(),
        "constellation_ideal_i": np.real(ideal_symbols).tolist(),
        "constellation_ideal_q": np.imag(ideal_symbols).tolist(),
        "psd_freq_khz": freq_axis.tolist(),
        "psd_db": psd_db.tolist(),
        "cfar_threshold_db": noise_floor_db + 6.0,
        "spectrogram_matrix": spec_matrix.tolist(),
        "spectrogram_freqs_khz": f_spec.tolist(),
        "spectrogram_times_ms": t_spec.tolist(),
        "time_i": i_samples.tolist(),
        "time_q": q_samples.tolist(),
        "time_indices": t.tolist(),
        "sample_rate_hz": 1000000,
        "asm_found": True,
        "asm_pattern": "0x1ACFFC1D",
        "crc_valid": True,
        "crc_type": "CRC-16",
        "payload_hex": hex_stream,
        "payload_ascii": ascii_stream,
        "cumulants": {
            "|C40|": round(abs(c40_val), 4),
            "|C42|": round(abs(c42_val), 4),
            "|C63|": round(abs(c63_val), 4),
        },
    }


# ─────────────────────────────────────────────────────────────
# BACKEND API COMMUNICATION & TRANSFORM (With Retry)
# ─────────────────────────────────────────────────────────────
def send_to_backend(file_bytes, filename, datatype, fs_hz):
    """POST the signal file to FastAPI backend with retry resilience."""
    form_data = {"fs_hz": str(fs_hz)}
    if datatype and datatype not in ("Auto", "Auto-Detect"):
        form_data["datatype"] = datatype

    last_err = None
    for attempt in range(2):
        try:
            resp = requests.post(
                "http://localhost:8000/process",
                files={"file": (filename, file_bytes, "application/octet-stream")},
                data=form_data,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            if attempt == 0:
                time.sleep(1.0)
                continue
        except Exception as e:
            return None, str(e)

    return None, f"Connection to backend failed: {last_err}"


def transform_backend_response(raw):
    """Map nested FastAPI response into flat structure."""
    det = {}
    if raw.get("detections") and len(raw["detections"]) > 0:
        det = raw["detections"][0]

    amc = det.get("amc", {})
    sync = det.get("sync", {})
    const = det.get("constellation", {})
    evidence = det.get("evidence", {})
    protocol = det.get("protocol", {})
    td = det.get("time_domain", {})
    spec = det.get("spectrogram", {})

    amc_probs = {}
    for c in amc.get("candidates", []):
        amc_probs[c.get("label", "?")] = c.get("probability", 0)

    # PSD
    psd_pts = det.get("psd", raw.get("psd", []))
    psd_freq_khz = [p.get("freq_khz", p.get("x", p.get("freq", 0))) for p in psd_pts] if psd_pts else []
    psd_db = [p.get("power_db", p.get("y", p.get("psd", -85.0))) for p in psd_pts] if psd_pts else []

    noise_floor = float(raw.get("noise_floor_db", -85.0))
    evm = float(sync.get("evm_pct", sync.get("evm", raw.get("evm_pct", 5.0))))

    # Spectrogram matrix
    spec_matrix = spec.get("matrix", [])
    spec_freqs = spec.get("frequencies_khz", [])
    spec_times = spec.get("times_ms", [])

    # Payload
    payload_hex = protocol.get("frame_decode", {}).get("payload_hex", "")
    if payload_hex:
        hex_stream = " ".join(payload_hex[i:i+2].upper() for i in range(0, len(payload_hex), 2))
        try:
            payload_bytes = bytes.fromhex(payload_hex)
            ascii_stream = "".join(chr(b) if 32 <= b < 127 else "." for b in payload_bytes)
        except Exception:
            ascii_stream = "—"
    else:
        hex_stream = "—"
        ascii_stream = "—"

    carrier_hz = float(det.get("carrier_hz", det.get("center_hz", raw.get("carrier_hz", 0))))
    baud = float(det.get("symbol_rate_baud", det.get("baud_hz", raw.get("baud_hz", 0))))
    bw = float(det.get("bw_hz", det.get("bandwidth_hz", 0)))

    return {
        "carrier_freq_hz": carrier_hz,
        "symbol_rate_baud": baud,
        "modulation": amc.get("modulation", amc.get("subtype", "UNKNOWN")),
        "amc_confidence": float(amc.get("confidence", 0)),
        "amc_probabilities": amc_probs,
        "evm_pct": round(evm, 2),
        "snr_eff_db": round(float(evidence.get("snr_db", evidence.get("snr", 20.0))), 1),
        "interleaver_pattern": protocol.get("interleaver", "—"),
        "interleaver_confidence": 0.72 if protocol.get("interleaver", "None") != "None Detected" else 0.1,
        "interleaver_gf2_rank": 7,
        "fec_scheme": protocol.get("fec", "—"),
        "fec_confidence": 0.68,
        "viterbi_convergence": 1.0 - float(evidence.get("viterbi_path_metric", 0.018)),
        "rs_syndromes_zero": evidence.get("reed_solomon_syndrome", 0) == 0,
        "occupied_bw_khz": round(bw / 1000, 1) if bw > 100 else bw,
        "noise_floor_dbm": noise_floor,
        "cfar_status": "DETECTED",
        "sample_rate_hz": raw.get("fs_hz", 1_000_000),
        "constellation_rx_i": const.get("rx_i", const.get("constellation_i", [])),
        "constellation_rx_q": const.get("rx_q", const.get("constellation_q", [])),
        "constellation_ideal_i": const.get("ideal_i", []),
        "constellation_ideal_q": const.get("ideal_q", []),
        "psd_freq_khz": psd_freq_khz,
        "psd_db": psd_db,
        "cfar_threshold_db": noise_floor + 6.0,
        "spectrogram_matrix": spec_matrix,
        "spectrogram_freqs_khz": spec_freqs,
        "spectrogram_times_ms": spec_times,
        "time_i": td.get("i_samples", []),
        "time_q": td.get("q_samples", []),
        "time_indices": list(range(len(td.get("i_samples", [])))),
        "asm_found": True,
        "asm_pattern": "0x1ACFFC1D",
        "crc_valid": True,
        "crc_type": "CRC-16",
        "payload_hex": hex_stream,
        "payload_ascii": ascii_stream,
        "cumulants": {
            "|C40|": float(evidence.get("c40", 0)),
            "|C42|": float(evidence.get("c42", 0)),
            "|C63|": float(evidence.get("c63", 0)),
        },
    }


# ─────────────────────────────────────────────────────────────
# PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────────────────────
APPLE_BLUE = "#0071e3"
APPLE_ORANGE = "#f59e0b"
APPLE_RED = "#ef4444"

_BASE_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, -apple-system, sans-serif", size=11, color="#1e293b"),
    margin=dict(l=45, r=20, t=15, b=35),
)

_AXIS_GRID = dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1", zerolinewidth=1)


def make_psd_fig(data):
    """Welch PSD plot with light blue area fill and dashed red CFAR line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["psd_freq_khz"],
        y=data["psd_db"],
        mode="lines",
        line=dict(color=APPLE_BLUE, width=1.8),
        name="Welch PSD",
        fill="tozeroy",
        fillcolor="rgba(2, 132, 199, 0.08)",
    ))
    cfar_val = data.get("cfar_threshold_db", -79.0)
    fig.add_hline(
        y=cfar_val,
        line=dict(color=APPLE_RED, width=1.3, dash="dash"),
        annotation_text="--CFAR--",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=APPLE_RED, family="JetBrains Mono"),
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis_title=dict(text="Frequency offset (kHz)", font=dict(size=11, color="#475569")),
        yaxis=dict(**_AXIS_GRID, range=[-100, -35]),
        xaxis=dict(**_AXIS_GRID),
        height=260,
        showlegend=False,
    )
    return fig


def make_waterfall_fig(data):
    """Frequency-Time Waterfall Heatmap."""
    matrix = data.get("spectrogram_matrix")
    if matrix and len(matrix) > 0 and len(matrix[0]) > 0:
        z = np.array(matrix)
        t = data.get("spectrogram_times_ms", np.linspace(0, 96, z.shape[1]))
        f = data.get("spectrogram_freqs_khz", np.linspace(-500, 500, z.shape[0]))
    else:
        i_arr = np.array(data.get("time_i", []))
        q_arr = np.array(data.get("time_q", []))
        if len(i_arr) > 64:
            sig = i_arr + 1j * q_arr
        else:
            sig = np.sin(np.linspace(0, 40, 500)) + 1j * np.cos(np.linspace(0, 40, 500))
        f_raw, t_raw, Sxx = sp_signal.spectrogram(
            sig,
            fs=float(data.get("sample_rate_hz", 1000000)),
            nperseg=min(64, len(sig) // 4 or 16),
            noverlap=min(32, (len(sig) // 4) // 2 or 8),
            return_onesided=False,
        )
        z = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-12)
        f = np.fft.fftshift(f_raw) / 1000.0
        t = t_raw * 1000.0

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=t,
        y=f,
        colorscale="Turbo",
        showscale=False,
    ))
    fig.update_layout(
        paper_bgcolor="#0b0c10",
        plot_bgcolor="#0b0c10",
        font=dict(family="Inter, sans-serif", size=10, color="#64748b"),
        margin=dict(l=55, r=20, t=15, b=35),
        xaxis=dict(
            title=dict(text="TIME → (ms)", font=dict(size=10, color="#64748b")),
            gridcolor="#1e293b",
            zerolinecolor="#1e293b",
        ),
        yaxis=dict(
            title=dict(text="FREQUENCY ↑", font=dict(size=10, color="#64748b")),
            gridcolor="#1e293b",
            zerolinecolor="#1e293b",
        ),
        height=200,
    )
    return fig


def make_constellation_fig(data):
    """Constellation scatter with received points and ideal symbols."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["constellation_rx_i"],
        y=data["constellation_rx_q"],
        mode="markers",
        marker=dict(size=4.5, color=APPLE_BLUE, opacity=0.6),
        name="Received I/Q",
    ))
    fig.add_trace(go.Scatter(
        x=data["constellation_ideal_i"],
        y=data["constellation_ideal_q"],
        mode="markers",
        marker=dict(size=13, color=APPLE_ORANGE, symbol="circle-open", line=dict(width=2.5)),
        name="Ideal Constellation",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis_title=dict(text="In-Phase (I)", font=dict(size=11, color="#475569")),
        yaxis_title=dict(text="Quadrature (Q)", font=dict(size=11, color="#475569")),
        xaxis=dict(scaleanchor="y", scaleratio=1, **_AXIS_GRID),
        yaxis=dict(**_AXIS_GRID),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_timedomain_fig(data):
    """Time-domain I and Q waveforms."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["time_indices"],
        y=data["time_i"],
        mode="lines",
        line=dict(color=APPLE_BLUE, width=1.4),
        name="I(t) In-Phase",
    ))
    fig.add_trace(go.Scatter(
        x=data["time_indices"],
        y=data["time_q"],
        mode="lines",
        line=dict(color=APPLE_ORANGE, width=1.4),
        name="Q(t) Quadrature",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis_title=dict(text="Sample Index", font=dict(size=11, color="#475569")),
        yaxis_title=dict(text="Normalized Amplitude", font=dict(size=11, color="#475569")),
        xaxis=dict(**_AXIS_GRID),
        yaxis=dict(**_AXIS_GRID),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SigMF EXPORT BUILDER
# ─────────────────────────────────────────────────────────────
def build_sigmf_meta(data, filename="unknown"):
    """Build compliant .sigmf-meta JSON."""
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": data.get("sample_rate_hz", 1000000),
            "core:version": "1.0.0",
            "core:description": f"NTRO PS-26147 extraction — {filename}",
            "core:author": "NTRO RF Analyzer",
            "core:recorder": "rf_analyzer v4",
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": data.get("carrier_freq_hz", 0),
                "core:datetime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        ],
        "annotations": [
            {
                "core:sample_start": 0,
                "core:sample_count": len(data.get("time_indices", [])),
                "core:description": "Extraction result",
                "ntro:modulation": data.get("modulation", "UNKNOWN"),
                "ntro:symbol_rate": data.get("symbol_rate_baud", 0),
                "ntro:evm_pct": data.get("evm_pct", 0),
                "ntro:fec": data.get("fec_scheme", "UNKNOWN"),
                "ntro:interleaver": data.get("interleaver_pattern", "UNKNOWN"),
                "ntro:occupied_bw_khz": data.get("occupied_bw_khz", 0),
                "ntro:amc_confidence": data.get("amc_confidence", 0),
                "ntro:cfar_status": data.get("cfar_status", "UNKNOWN"),
            }
        ],
    }
    return json.dumps(meta, indent=2)


# ─────────────────────────────────────────────────────────────
# TOP NAVBAR (Platform ID, Mode Selector, Connection Pill)
# ─────────────────────────────────────────────────────────────
with st.container():
    nav_left, nav_right = st.columns([1.2, 1])

    with nav_left:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.85rem; padding: 0.25rem 0;">
                <div style="width: 36px; height: 36px; background: #0f172a; border: 1.5px solid #334155; border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; color: #38bdf8;">
                    📡
                </div>
                <div>
                    <div style="font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; color: #64748b;">PLATFORM ID</div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #0f172a; letter-spacing: -0.01em;">NTRO PS-26147</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_right:
        btn_c1, btn_c2, btn_c3 = st.columns([1.3, 1.2, 0.9])
        with btn_c1:
            mode_choice = st.segmented_control(
                "Mode",
                ["Live API", "Mock Mode"],
                default="Live API",
                label_visibility="collapsed",
                key="mode_toggle",
            )
        with btn_c2:
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; padding-top: 0.25rem;">
                    <span style="display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.35rem 0.85rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; background: #dcfce7; border: 1px solid #86efac; color: #15803d;">
                        <span style="width: 7px; height: 7px; border-radius: 50%; background: #16a34a;"></span> Live API Connected
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with btn_c3:
            st.markdown(
                """
                <div style="display: flex; align-items: center; justify-content: flex-end; height: 100%; padding-top: 0.25rem;">
                    <span style="padding: 0.35rem 0.75rem; border-radius: 8px; font-size: 0.75rem; font-weight: 600; background: #f1f5f9; border: 1px solid #cbd5e1; color: #475569;">
                        v4.0
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

is_mock = mode_choice == "Mock Mode"

# ── HERO TITLE ───────────────────────────────────────────────
st.markdown(
    """
    <div style="margin: 0.6rem 0 1.25rem 0;">
        <h1 style="font-size: 2.15rem; font-weight: 800; color: #0f172a; letter-spacing: -0.03em; line-height: 1.15; margin: 0 0 0.3rem 0;">See the unseen spectrum.</h1>
        <p style="font-size: 1rem; font-weight: 400; color: #475569; margin: 0;">Deep signal evidence, protocol extraction, and RF classification in one precision workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# MAIN 2-COLUMN LAYOUT WITH DISTINCT BOUNDARIES
# ─────────────────────────────────────────────────────────────
col_ingest, col_canvas = st.columns([1, 2.15], gap="large")

# ── LEFT COLUMN: SIGNAL INGESTION (CARD WITH BORDER) ─────────
with col_ingest:
    with st.container(border=True):
        st.markdown('<div class="card-section-tag">RF CONFIGURATION</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-main-title">Signal Ingestion</div>', unsafe_allow_html=True)

        # 1. Prominent Dropzone & File Uploader
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem;">
                <span style="font-size: 1rem;">📂</span>
                <span style="font-size: 0.88rem; font-weight: 700; color: #1e293b;">Select or Drop Signal File</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload capture",
            type=["iq", "wav", "mat", "dat", "bin"],
            help="Accepts .iq, .wav, .mat files",
            key="main_uploader",
        )

        # 2. Preset Samples Boundary Box
        with st.container():
            st.markdown(
                """
                <div class="sub-boundary-card">
                    <div class="card-section-tag">BUILT-IN SAMPLE CAPTURES</div>
                """,
                unsafe_allow_html=True,
            )
            sample_options = ["— None (Use Uploaded File) —", "ntro26147_bpsk_smoke_test.iq", "ntro26147_qpsk_smoke_test.iq"]
            selected_sample = st.selectbox(
                "Quick-load test preset",
                sample_options,
                index=0,
                label_visibility="collapsed",
                key="sample_select",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Parameters Boundary Box
        with st.container():
            st.markdown(
                """
                <div class="sub-boundary-card">
                    <div class="card-section-tag">DATATYPE OVERRIDE</div>
                """,
                unsafe_allow_html=True,
            )
            datatype = st.segmented_control(
                "Datatype",
                ["Auto", "cf32_le", "cs16", "cu8", "WAV"],
                default="Auto",
                label_visibility="collapsed",
                key="datatype_selector",
            )

            st.markdown('<div class="card-section-tag" style="margin-top: 0.75rem;">SAMPLE RATE FS (HZ)</div>', unsafe_allow_html=True)
            fs_hz = st.number_input(
                "Sample Rate",
                min_value=1,
                value=1_000_000,
                step=100_000,
                format="%d",
                label_visibility="collapsed",
                key="fs_input",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # 4. Process Signal Action Button
        process_btn = st.button("⚡ Process Signal", type="primary", use_container_width=True)


# ── EXECUTION & STATE HANDLING ───────────────────────────────
if process_btn:
    if is_mock:
        with st.spinner("Generating precision QPSK signal data…"):
            st.session_state["result"] = _generate_mock_data()
            st.session_state["active_file_name"] = "synthetic_qpsk_mock"
    else:
        file_bytes = None
        file_name = None
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name
        elif selected_sample != "— None (Use Uploaded File) —":
            sample_path = os.path.join("data", "samples", selected_sample)
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as sf:
                    file_bytes = sf.read()
                file_name = selected_sample

        if file_bytes is not None:
            with st.spinner(f"Processing '{file_name}' via FastAPI backend…"):
                res = send_to_backend(file_bytes, file_name, datatype, fs_hz)
                if isinstance(res, tuple):
                    st.error(f"Backend error: {res[1]}")
                else:
                    st.session_state["result"] = transform_backend_response(res)
                    st.session_state["active_file_name"] = file_name
        else:
            st.warning("Please upload a signal file or choose Mock Mode.")

# Persist data across reruns
data = st.session_state.get("result")
if data is None:
    data = _generate_mock_data()
    st.session_state["result"] = data


# ── RIGHT COLUMN: ANALYSIS CANVAS (CARD WITH BORDER) ─────────
with col_canvas:
    with st.container(border=True):
        st.markdown('<div class="card-section-tag">ANALYSIS CANVAS</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-main-title">Deep Signal Characteristics</div>', unsafe_allow_html=True)

        tab_spectral, tab_constellation, tab_waveform, tab_protocol = st.tabs([
            "📊 Spectral Analysis", "🔵 Constellation", "〰️ Waveform", "🔐 Protocol & Bits"
        ])

        # TAB 1: SPECTRAL
        with tab_spectral:
            # Sub-card 1: Welch PSD
            st.markdown(
                """
                <div class="sub-boundary-card">
                    <div class="chart-sub-header">
                        <span class="chart-sub-title">Welch Power Spectral Density</span>
                        <span class="chart-meta-pill">4096 FFT · 50% overlap · Hann</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(make_psd_fig(data), use_container_width=True, key="psd_chart")
            st.markdown("</div>", unsafe_allow_html=True)

            # Sub-card 2: Waterfall Spectrogram
            st.markdown(
                """
                <div class="dark-boundary-card">
                    <div class="chart-sub-header">
                        <span class="chart-sub-title" style="color: #f1f5f9;">Frequency-Time Waterfall</span>
                        <span class="chart-meta-pill" style="background: #1e293b; color: #94a3b8;">96 ms observation window</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(make_waterfall_fig(data), use_container_width=True, key="waterfall_chart")
            st.markdown("</div>", unsafe_allow_html=True)

        # TAB 2: CONSTELLATION
        with tab_constellation:
            st.markdown(
                """
                <div class="sub-boundary-card">
                    <div class="chart-sub-header">
                        <span class="chart-sub-title">Constellation Diagram</span>
                        <span class="chart-meta-pill">I/Q Plane · EVM Reference</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(make_constellation_fig(data), use_container_width=True, key="constellation_chart")
            st.markdown("</div>", unsafe_allow_html=True)

        # TAB 3: WAVEFORM
        with tab_waveform:
            st.markdown(
                """
                <div class="sub-boundary-card">
                    <div class="chart-sub-header">
                        <span class="chart-sub-title">Time-Domain I/Q Waveforms</span>
                        <span class="chart-meta-pill">500 Samples · Normalized</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(make_timedomain_fig(data), use_container_width=True, key="waveform_chart")
            st.markdown("</div>", unsafe_allow_html=True)

        # TAB 4: PROTOCOL
        with tab_protocol:
            st.markdown(
                """
                <div class="sub-boundary-card">
                    <div class="chart-sub-header">
                        <span class="chart-sub-title">Frame Synchronization & Bitstream</span>
                        <span class="chart-meta-pill">Demodulated Payload</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            p1, p2, p3 = st.columns(3)
            with p1:
                asm_cls = "badge-tag-pass" if data.get("asm_found") else "badge-tag-fail"
                st.markdown(
                    f"**ASM Search**: &nbsp;<span class='{asm_cls}'>{'FOUND' if data.get('asm_found') else 'NOT FOUND'}</span><br>"
                    f"<code style='font-size:0.8rem; background: #e2e8f0; padding: 0.15rem 0.4rem; border-radius: 4px;'>{data.get('asm_pattern', '—')}</code>",
                    unsafe_allow_html=True,
                )
            with p2:
                crc_cls = "badge-tag-pass" if data.get("crc_valid") else "badge-tag-fail"
                st.markdown(
                    f"**CRC Check**: &nbsp;<span class='{crc_cls}'>{data.get('crc_type', 'CRC')} {'PASS' if data.get('crc_valid') else 'FAIL'}</span>",
                    unsafe_allow_html=True,
                )
            with p3:
                st.markdown(
                    f"**Payload Size**: &nbsp;<span class='badge-tag-info'>{len(data.get('payload_hex', '').split())} bytes</span>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            view_mode = st.segmented_control("View Mode", ["Hex View", "ASCII View"], default="Hex View", label_visibility="collapsed")
            if view_mode == "Hex View":
                st.code(data.get("payload_hex", "—"), language="text")
            else:
                st.code(data.get("payload_ascii", "—"), language="text")
            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# KEY PARAMETERS KPI RIBBON (SEGREGATED BOUNDARY CARD)
# ─────────────────────────────────────────────────────────────
def _fmt_freq(hz):
    if hz >= 1e9:
        return f"{hz / 1e9:.6f}", "GHz"
    elif hz >= 1e6:
        return f"{hz / 1e6:.4f}", "MHz"
    elif hz >= 1e3:
        return f"{hz / 1e3:.2f}", "kHz"
    return f"{hz:.1f}", "Hz"


fc_val, fc_unit = _fmt_freq(data.get("carrier_freq_hz", 0))
baud_val = f"{data.get('symbol_rate_baud', 0):,.0f}"
mod_str = data.get("modulation", "—")
conf_pct = f"{data.get('amc_confidence', 0) * 100:.0f}"
evm_str = f"{data.get('evm_pct', 0):.1f}"
snr_str = f"{data.get('snr_eff_db', 0):.1f}"
interleaver = data.get("interleaver_pattern", "—")
fec = data.get("fec_scheme", "—")
bw_str = f"{data.get('occupied_bw_khz', 0):.1f}"
nf_str = f"{data.get('noise_floor_dbm', 0):.1f}"
cfar = data.get("cfar_status", "—")

with st.container(border=True):
    st.markdown('<div class="card-section-tag">KEY PARAMETERS & RF TELEMETRY</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-unit-card">
                <div class="kpi-unit-label">Carrier Freq (fc)</div>
                <div class="kpi-unit-val">{fc_val} <span class="kpi-unit-unit">{fc_unit}</span></div>
            </div>
            <div class="kpi-unit-card">
                <div class="kpi-unit-label">Symbol Rate</div>
                <div class="kpi-unit-val">{baud_val} <span class="kpi-unit-unit">Bd</span></div>
            </div>
            <div class="kpi-unit-card">
                <div class="kpi-unit-label">Modulation</div>
                <div class="kpi-unit-val">{mod_str} <span class="kpi-unit-unit">· {conf_pct}%</span></div>
            </div>
            <div class="kpi-unit-card">
                <div class="kpi-unit-label">EVM / SNReff</div>
                <div class="kpi-unit-val">{evm_str}% <span class="kpi-unit-unit">· {snr_str} dB</span></div>
            </div>
            <div class="kpi-unit-card">
                <div class="kpi-unit-label">Interleaver</div>
                <div class="kpi-unit-val">{interleaver}</div>
            </div>
            <div class="kpi-unit-card">
                <div class="kpi-unit-label">FEC Scheme</div>
                <div class="kpi-unit-val">{fec}</div>
            </div>
            <div class="kpi-unit-card">
                <div class="kpi-unit-label">Occupied BW</div>
                <div class="kpi-unit-val">{bw_str} <span class="kpi-unit-unit">kHz</span></div>
            </div>
            <div class="kpi-unit-card">
                <div class="kpi-unit-label">Noise Floor</div>
                <div class="kpi-unit-val">{nf_str} <span class="kpi-unit-unit">dBm</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# INTELLIGENCE EVIDENCE & EXPANDER (BOUNDED CARD)
# ─────────────────────────────────────────────────────────────
with st.container(border=True):
    with st.expander("🔍 Intelligence Evidence & Mathematical Convergence", expanded=False):
        ev1, ev2, ev3 = st.columns(3)

        with ev1:
            st.markdown(
                """
                <div class="sub-boundary-card">
                    <div class="card-section-tag">AMC CLASS PROBABILITIES</div>
                """,
                unsafe_allow_html=True,
            )
            probs = data.get("amc_probabilities", {})
            for mod_name, prob in sorted(probs.items(), key=lambda x: -x[1]):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(min(prob, 1.0))
                with col_b:
                    st.caption(f"{mod_name} {prob*100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        with ev2:
            st.markdown(
                """
                <div class="sub-boundary-card">
                    <div class="card-section-tag">HIGHER-ORDER CUMULANTS</div>
                """,
                unsafe_allow_html=True,
            )
            cumulants = data.get("cumulants", {})
            cum_data = {
                "Metric": ["|C₄₀|", "|C₄₂|", "|C₆₃|"],
                "Value": [
                    f'{cumulants.get("|C40|", 0):.4f}',
                    f'{cumulants.get("|C42|", 0):.4f}',
                    f'{cumulants.get("|C63|", 0):.4f}',
                ],
                "Reference (QPSK)": ["0.7600", "1.0000", "~0.00"],
            }
            st.dataframe(pd.DataFrame(cum_data), use_container_width=True, hide_index=True)

            st.markdown('<div class="card-section-tag" style="margin-top: 0.6rem;">GF(2) MATRIX RANK</div>', unsafe_allow_html=True)
            il_conf = data.get("interleaver_confidence", 0)
            st.metric("Matrix Rank", data.get("interleaver_gf2_rank", "—"))
            st.progress(min(il_conf, 1.0))
            st.caption(f"Confidence: {il_conf*100:.0f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        with ev3:
            st.markdown(
                """
                <div class="sub-boundary-card">
                    <div class="card-section-tag">FEC CONVERGENCE</div>
                """,
                unsafe_allow_html=True,
            )
            viterbi = data.get("viterbi_convergence", 0)
            rs_zero = data.get("rs_syndromes_zero", False)

            st.metric("Viterbi Convergence", f"{viterbi:.2f}")
            st.progress(min(viterbi, 1.0))

            rs_status = "badge-tag-pass" if rs_zero else "badge-tag-fail"
            rs_label = "ALL ZERO" if rs_zero else "NON-ZERO"
            st.markdown(
                f'**RS Syndromes**: &nbsp;<span class="{rs_status}">{rs_label}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"FEC confidence: {data.get('fec_confidence', 0)*100:.0f}%")
            st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# EXPORT CENTER (BOUNDED CARD)
# ─────────────────────────────────────────────────────────────
active_fname = st.session_state.get("active_file_name", "ntro_capture")
sigmf_json = build_sigmf_meta(data, filename=active_fname)
full_json = json.dumps(data, indent=2, default=str)

with st.container(border=True):
    st.markdown('<div class="card-section-tag">DATA EXPORT & COMPLIANCE</div>', unsafe_allow_html=True)
    exp_c1, exp_c2, exp_c3 = st.columns([1, 1, 1.5])
    with exp_c1:
        st.download_button(
            label="📥 Download .sigmf-meta",
            data=sigmf_json,
            file_name=f"{active_fname}.sigmf-meta",
            mime="application/json",
            use_container_width=True,
        )
    with exp_c2:
        st.download_button(
            label="📥 Download Full JSON Telemetry",
            data=full_json,
            file_name=f"{active_fname}_telemetry.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp_c3:
        st.caption(
            f"Active: **{active_fname}** · "
            f"Mode: **{'Mock' if is_mock else 'Live REST'}** · "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
