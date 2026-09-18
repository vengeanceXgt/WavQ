"""Streamlit GUI: load a file, run the pipeline, show plots and confidence per stage."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from rf_analyzer.ingest.characterize import load_file, normalize
from rf_analyzer.spectral.features import power_spectral_density
from rf_analyzer.orchestrator.pipeline import run_pipeline

st.set_page_config(page_title="RF Signal Analyzer", layout="wide")
st.title("Automated RF Signal Analysis Workstation")

uploaded = st.file_uploader("Load a .iq or .wav file", type=["iq", "wav", "dat", "bin"])
sample_rate_override = st.number_input("Manual sample rate override (Hz, optional)",
                                        min_value=0.0, value=0.0)

if uploaded is not None:
    tmp_path = f"data/samples/{uploaded.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    overrides = {}
    if sample_rate_override > 0:
        overrides["sample_rate"] = sample_rate_override

    stages = run_pipeline(tmp_path, assumed_sample_rate=sample_rate_override or None,
                           manual_overrides=overrides)

    ingest_stage = next((s for s in stages if s.name == "ingest"), None)
    if ingest_stage and ingest_stage.ok:
        result = load_file(tmp_path, assumed_sample_rate=sample_rate_override or None)
        iq, _ = normalize(result.iq)
        sr = sample_rate_override or result.sample_rate or 1.0

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Time-Domain Envelope")
            fig = go.Figure(go.Scatter(y=np.abs(iq[:2000])))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Power Spectral Density")
            freqs, psd = power_spectral_density(iq, sr)
            fig = go.Figure(go.Scatter(x=freqs, y=10 * np.log10(psd + 1e-15)))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pipeline Results")
    for s in stages:
        status = "PASS" if s.ok else "FAIL"
        with st.expander(f"{status} {s.name}  (confidence: {s.confidence})"):
            st.json(s.data if s.ok else {"error": s.error})
else:
    st.info("Upload a .iq or .wav file to begin analysis.")
