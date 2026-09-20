import numpy as np
import plotly.graph_objects as go
from rf_analyzer.spectral.features import power_spectral_density

def plot_waveform(iq, max_samples=2000):
    fig = go.Figure(go.Scatter(y=np.abs(iq[:max_samples]), mode='lines', name='Envelope'))
    fig.update_layout(title="Time-Domain Envelope", xaxis_title="Sample", yaxis_title="Magnitude")
    return fig

def plot_psd(iq, sample_rate):
    freqs, psd = power_spectral_density(iq, sample_rate)
    fig = go.Figure(go.Scatter(x=freqs, y=10 * np.log10(psd + 1e-15), mode='lines'))
    fig.update_layout(title="Power Spectral Density", xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)")
    return fig

def plot_constellation(symbols, max_symbols=2000):
    syms = symbols[:max_symbols]
    fig = go.Figure(go.Scatter(x=np.real(syms), y=np.imag(syms), mode='markers', 
                               marker=dict(size=4, opacity=0.6)))
    fig.update_layout(title="Symbol Constellation", xaxis_title="In-Phase", yaxis_title="Quadrature",
                      width=500, height=500)
    # Ensure square aspect ratio
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig
