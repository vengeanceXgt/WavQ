import numpy as np
import plotly.graph_objects as go
from rf_analyzer.spectral.features import power_spectral_density

def validate_plot_data(x, y=None):
    if x is None:
        return False
    if y is not None:
        if len(x) != len(y):
            return False
        if len(y) == 0:
            return False
        if not np.isrealobj(y) and not np.isrealobj(x):
            pass # allow complex scatter but we usually split it
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            return False
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return False
    else:
        if len(x) == 0:
            return False
        if not np.iscomplexobj(x) and not np.isrealobj(x):
            return False
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return False
    return True

def generate_unavailable_plot(title, reason):
    fig = go.Figure()
    fig.add_annotation(
        text=reason,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="red")
    )
    fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

def plot_waveform(iq, max_samples=2000):
    if not validate_plot_data(iq):
        return generate_unavailable_plot("Time-Domain Envelope", "Signal trace unavailable (Invalid Data)")

    y_data = np.abs(iq[:max_samples])
    if not validate_plot_data(y_data):
        return generate_unavailable_plot("Time-Domain Envelope", "Signal trace unavailable (NaN/Inf values)")

    fig = go.Figure(go.Scatter(y=y_data, mode='lines', name='Envelope'))
    fig.update_layout(title="Time-Domain Envelope", xaxis_title="Sample", yaxis_title="Magnitude")
    return fig

def plot_psd(iq, sample_rate):
    if not validate_plot_data(iq):
        return generate_unavailable_plot("Power Spectral Density", "Spectrum unavailable (Invalid Data)")

    try:
        freqs, psd = power_spectral_density(iq, sample_rate)
        psd_db = 10 * np.log10(psd + 1e-15)
        if not validate_plot_data(freqs, psd_db):
            return generate_unavailable_plot("Power Spectral Density", "Spectrum unavailable (NaN/Inf in spectrum calculation)")

        fig = go.Figure(go.Scatter(x=freqs, y=psd_db, mode='lines'))
        fig.update_layout(title="Power Spectral Density", xaxis_title="Relative Frequency (Hz)", yaxis_title="Power (dB)")
        return fig
    except Exception as e:
        return generate_unavailable_plot("Power Spectral Density", f"Spectrum unavailable: {e}")

def plot_constellation(symbols, max_symbols=2000):
    if not validate_plot_data(symbols):
        return generate_unavailable_plot("Symbol Constellation", "Symbol recovery did not produce a valid symbol stream.")

    syms = symbols[:max_symbols]
    x_data = np.real(syms)
    y_data = np.imag(syms)

    if not validate_plot_data(x_data, y_data):
        return generate_unavailable_plot("Symbol Constellation", "Symbol recovery produced invalid (NaN/Inf) symbols.")

    fig = go.Figure(go.Scatter(x=x_data, y=y_data, mode='markers',
                               marker=dict(size=4, opacity=0.6)))
    fig.update_layout(title="Symbol Constellation", xaxis_title="In-Phase", yaxis_title="Quadrature",
                      width=500, height=500)
    # Ensure square aspect ratio
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig

def plot_cfo_trajectory(cfo_array):
    if cfo_array is None:
         return generate_unavailable_plot("CFO Trajectory", "CFO Trajectory unavailable (No data)")
    if isinstance(cfo_array, (float, int)):
         # It's a scalar
         fig = go.Figure()
         fig.add_annotation(
             text=f"Scalar CFO Estimate: {cfo_array:.2f} Hz",
             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
             font=dict(size=14, color="white")
         )
         fig.update_layout(title="CFO Trajectory", xaxis=dict(visible=False), yaxis=dict(visible=False))
         return fig

    if not validate_plot_data(cfo_array):
         return generate_unavailable_plot("CFO Trajectory", "CFO trajectory contains invalid data.")

    fig = go.Figure(go.Scatter(y=cfo_array, mode='lines'))
    fig.update_layout(title="CFO Trajectory", xaxis_title="Window", yaxis_title="CFO (Hz)")
    return fig

def plot_frame_correlation(corr_array):
    if not validate_plot_data(corr_array):
        return generate_unavailable_plot("Frame Correlation", "FRAME STRUCTURE NOT IDENTIFIED")

    fig = go.Figure(go.Scatter(y=corr_array, mode='lines'))
    fig.update_layout(title="Frame Correlation", xaxis_title="Lag", yaxis_title="Autocorrelation")
    return fig
