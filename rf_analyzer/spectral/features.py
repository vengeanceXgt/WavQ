"""Stages 2-6: Signal Detection, Noise Estimation, Carrier/Bandwidth/Symbol-rate estimation."""
import numpy as np
from scipy.signal import welch

def power_spectral_density(iq, sample_rate, nperseg=1024):
    freqs, psd = welch(iq, fs=sample_rate, nperseg=min(nperseg, len(iq)),
                        return_onesided=False)
    order = np.argsort(freqs)
    return freqs[order], psd[order]

def estimate_noise_floor_db(psd_db, percentile=20):
    """Assumes most bins are noise; the low percentile approximates the noise floor."""
    return float(np.percentile(psd_db, percentile))

def detect_occupied_band(freqs, psd, threshold_db_above_floor=6.0):
    psd_db = 10 * np.log10(psd + 1e-15)
    floor = estimate_noise_floor_db(psd_db)
    mask = psd_db > (floor + threshold_db_above_floor)
    if not np.any(mask):
        return None  # no signal found above threshold
    active_freqs = freqs[mask]
    f_lo, f_hi = float(active_freqs.min()), float(active_freqs.max())
    bandwidth = f_hi - f_lo
    carrier = (f_hi + f_lo) / 2.0
    return {"carrier_hz": carrier, "bandwidth_hz": bandwidth,
            "f_lo": f_lo, "f_hi": f_hi, "noise_floor_db": floor}

def estimate_cfo_nonlinear(iq, sample_rate, power=4):
    """Nonlinear (Nth-power) spectral method: raises signal to a power that collapses
    PSK/QAM modulation, leaving a strong tone at N * carrier_offset."""
    raised = iq.astype(complex) ** power
    spectrum = np.fft.fftshift(np.fft.fft(raised))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1.0 / sample_rate))
    peak_idx = np.argmax(np.abs(spectrum))
    return float(freqs[peak_idx] / power)

def mix_to_baseband(iq, sample_rate, cfo_hz):
    t = np.arange(len(iq)) / sample_rate
    return (iq * np.exp(-1j * 2 * np.pi * cfo_hz * t)).astype(np.complex64)
