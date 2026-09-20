"""Stage 8 (part 2): Costas loop for residual carrier phase/frequency tracking."""
import numpy as np

def costas_loop_qpsk(symbols, loop_bw=0.02, damping=0.707):
    alpha = loop_bw
    beta = loop_bw ** 2 / damping
    phase, freq = 0.0, 0.0
    out = np.empty_like(symbols)
    for n, s in enumerate(symbols):
        corrected = s * np.exp(-1j * phase)
        out[n] = corrected
        # QPSK phase-error detector
        error = np.sign(corrected.real) * corrected.imag - np.sign(corrected.imag) * corrected.real
        freq += beta * error
        phase += freq + alpha * error
    return out


import numpy as np
from rf_analyzer.spectral.features import estimate_cfo_nonlinear

def track_cfo_blockwise(iq, sample_rate, window_size=8192, power=4, smooth_len=5):
    """
    Block-based continuous frequency tracking.
    Breaks the signal into windows, estimates CFO per window, smooths the trajectory,
    and applies continuous phase correction.
    
    Returns:
        corrected_iq: The frequency-corrected signal.
        cfo_trajectory: Array of CFO estimates per window.
    """
    num_windows = len(iq) // window_size
    if num_windows == 0:
        cfo = estimate_cfo_nonlinear(iq, sample_rate, power)
        t = np.arange(len(iq)) / sample_rate
        return (iq * np.exp(-1j * 2 * np.pi * cfo * t)).astype(np.complex64), np.array([cfo])
        
    cfo_estimates = np.zeros(num_windows)
    for i in range(num_windows):
        block = iq[i*window_size : (i+1)*window_size]
        cfo_estimates[i] = estimate_cfo_nonlinear(block, sample_rate, power)
        
    # Simple moving average smoothing
    if smooth_len > 1 and num_windows > smooth_len:
        smoothed = np.convolve(cfo_estimates, np.ones(smooth_len)/smooth_len, mode='valid')
        # Pad ends
        pad_start = (smooth_len - 1) // 2
        pad_end = smooth_len - 1 - pad_start
        cfo_estimates = np.pad(smoothed, (pad_start, pad_end), mode='edge')
        
    # Interpolate CFO for every sample
    t_windows = (np.arange(num_windows) * window_size + window_size / 2) / sample_rate
    t_samples = np.arange(len(iq)) / sample_rate
    
    # Linear interpolation of frequency
    inst_cfo = np.interp(t_samples, t_windows, cfo_estimates)
    
    # Phase is the integral of frequency: phase(t) = 2 * pi * int_0^t f(tau) d_tau
    # Since we have discrete samples, phase[n] = phase[n-1] + 2 * pi * f[n] / fs
    phase = 2 * np.pi * np.cumsum(inst_cfo) / sample_rate
    
    corrected_iq = (iq * np.exp(-1j * phase)).astype(np.complex64)
    
    return corrected_iq, cfo_estimates
