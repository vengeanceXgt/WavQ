import numpy as np
import pytest
from rf_analyzer.export.plots import plot_waveform, plot_psd, plot_constellation, plot_cfo_trajectory, plot_frame_correlation

def validate_plot_data(fig, expected_traces=1):
    assert fig is not None
    assert len(fig.data) == expected_traces, f"Expected {expected_traces} traces, got {len(fig.data)}"
    
    for trace in fig.data:
        x = trace.x
        y = trace.y
        
        # Plotly traces might have x=None if implicitly indexed (like in plot_waveform)
        if x is not None:
            assert len(x) == len(y), "x and y lengths do not match"
            
        assert len(y) > 0, "Trace contains no points"
        assert not np.any(np.isnan(y)), "Trace contains NaNs"
        assert not np.any(np.isinf(y)), "Trace contains Infs"

def test_envelope_plot():
    iq = np.random.randn(1000) + 1j * np.random.randn(1000)
    fig = plot_waveform(iq)
    validate_plot_data(fig)

def test_psd_plot():
    iq = np.random.randn(1000) + 1j * np.random.randn(1000)
    fig = plot_psd(iq, 1e6)
    validate_plot_data(fig)

def test_constellation_plot():
    syms = np.array([1+1j, 1-1j, -1+1j, -1-1j])
    fig = plot_constellation(syms)
    validate_plot_data(fig)

def test_cfo_trajectory_scalar():
    fig = plot_cfo_trajectory(123.45)
    assert fig is not None
    # Annotations should exist
    assert len(fig.layout.annotations) > 0

def test_cfo_trajectory_array():
    cfo = np.array([10, 15, 20])
    fig = plot_cfo_trajectory(cfo)
    validate_plot_data(fig)

def test_frame_correlation():
    corr = np.array([0.1, 0.9, 0.1])
    fig = plot_frame_correlation(corr)
    validate_plot_data(fig)
