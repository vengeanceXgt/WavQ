"""Streamlit GUI MVP for RF Signal Analyzer."""
import streamlit as st
import numpy as np
import tempfile
import os
import json

from rf_analyzer.orchestrator.pipeline import analyze_signal
from rf_analyzer.ingest.characterize import load_file, normalize
from rf_analyzer.export.plots import plot_waveform, plot_psd, plot_constellation
from rf_analyzer.sync.carrier import track_cfo_blockwise
from rf_analyzer.sync.matched_filter import apply_rx_matched_filter
from rf_analyzer.sync.timing import gardner_timing_recovery
from rf_analyzer.spectral.features import estimate_cfo_nonlinear, mix_to_baseband

st.set_page_config(page_title="RF Signal Analyzer MVP", layout="wide")
st.title("Automated RF Signal Analysis Workstation")

def display_status(status_str):
    color = "gray"
    if status_str in ["success", "detected", "identified", "locked", "complete"]:
        color = "green"
    elif status_str in ["ambiguous", "partial"]:
        color = "orange"
    elif status_str in ["failed"]:
        color = "red"
    return f"<span style='color:{color}; font-weight:bold;'>{status_str.upper()}</span>"

uploaded = st.file_uploader("Load a .iq or .wav file", type=["iq", "wav", "dat", "bin", "sigmf-data"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".iq") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name
        
    if st.button("Run Analysis"):
        with st.spinner("Analyzing Signal..."):
            result = analyze_signal(tmp_path)
            
        st.header("1. Overview")
        st.markdown(f"**Pipeline Status:** {display_status(result['status'])}", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2. Signal")
            st.json(result["signal"])
        with col2:
            st.subheader("3. Synchronization")
            st.json(result["synchronization"])
            
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("4. Modulation")
            mod = result["modulation"]
            st.markdown(f"**Label:** {mod['label'].upper()}")
            st.markdown(f"**Status:** {display_status('ambiguous' if mod.get('ambiguous') else 'supported')}", unsafe_allow_html=True)
            st.json(mod.get("candidates", []))
        with col4:
            st.subheader("5. Demodulation")
            st.json(result["demodulation"])
            
        st.subheader("6. Frame & FEC")
        col5, col6 = st.columns(2)
        with col5:
            st.json(result["frame"])
        with col6:
            st.json(result["fec"])
            
        if result["limitations"]:
            st.subheader("Limitations / Failures")
            for lim in result["limitations"]:
                st.error(lim)
                
        st.header("7. Visualizations")
        # Generate plots on the fly for the UI
        try:
            ingest = load_file(tmp_path)
            iq, _ = normalize(ingest.iq)
            sr = result["input"].get("sample_rate", 1.0)
            
            p1, p2 = st.columns(2)
            with p1:
                st.plotly_chart(plot_waveform(iq), use_container_width=True)
            with p2:
                st.plotly_chart(plot_psd(iq, sr), use_container_width=True)
                
            # If timing locked, show constellation
            if result["synchronization"].get("timing_status") == "locked":
                cfo = result["synchronization"].get("cfo_hz", estimate_cfo_nonlinear(iq, sr, 4))
                iq_bb = mix_to_baseband(iq, sr, cfo)
                sps = int(result["synchronization"].get("symbol_rate_sps", 4))
                cont = apply_rx_matched_filter(iq_bb, sps)
                syms = gardner_timing_recovery(cont, sps)
                st.plotly_chart(plot_constellation(syms), use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate all plots: {e}")
            
        st.header("8. Report")
        report_md = f"""# RF Analysis Report
## Input
File: {uploaded.name}
Samples: {result['input'].get('samples')}

## Signal
SNR (dB): {result['signal'].get('snr_db')}
Occupied BW (Hz): {result['signal'].get('occupied_bandwidth_hz')}

## Synchronization
CFO (Hz): {result['synchronization'].get('cfo_hz')}
Symbol Rate: {result['synchronization'].get('symbol_rate_sps')}
Timing: {result['synchronization'].get('timing_status')}

## Modulation
Candidate: {result['modulation'].get('label')}
Ambiguous: {result['modulation'].get('ambiguous')}

## Demodulation & FEC
Demod Status: {result['demodulation'].get('status')}
FEC Status: {result['fec'].get('status')}
Scheme: {result['fec'].get('scheme', 'none')}
"""
        st.markdown(report_md)
        st.download_button("Download Report", data=report_md, file_name="report.md")
        
        os.remove(tmp_path)
