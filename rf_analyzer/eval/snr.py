import numpy as np

class SNRResult:
    def __init__(self, snr_db, method, signal_power, noise_power, valid, reason=""):
        self.snr_db = snr_db
        self.method = method
        self.signal_power = signal_power
        self.noise_power = noise_power
        self.valid = valid
        self.reason = reason

def estimate_snr_m2m4(symbols):
    """
    Estimates SNR using the M2M4 algorithm, designed for M-PSK (and similar constant 
    or known amplitude modulation schemes) in AWGN.
    Assumes `symbols` is an array of complex baseband symbols.
    """
    if len(symbols) < 100:
        return SNRResult(None, "M2M4", None, None, False, "Insufficient symbols for M2M4 estimation")
        
    symbols = np.nan_to_num(symbols, nan=0.0, posinf=0.0, neginf=0.0)
    magnitude_sq = np.real(symbols)**2 + np.imag(symbols)**2
    
    m2 = np.mean(magnitude_sq)
    m4 = np.mean(magnitude_sq**2)
    
    val = 2 * (m2**2) - m4
    if val < 0:
        return SNRResult(None, "M2M4", None, None, False, "M2M4 mathematical constraint failed (negative variance)")
        
    s_est = np.sqrt(val)
    n_est = m2 - s_est
    
    if n_est <= 0:
        n_est = 1e-12
    if s_est <= 0:
        return SNRResult(None, "M2M4", float(s_est), float(n_est), False, "Signal power estimate is negative or zero")
        
    snr_db = 10 * np.log10(s_est / n_est)
    return SNRResult(float(snr_db), "M2M4", float(s_est), float(n_est), True)

def estimate_snr_spectral(iq, sample_rate, occupied_band=None):
    """
    Estimates SNR using Welch PSD by comparing power inside the occupied band 
    to power outside the occupied band (noise/guard bands).
    """
    if len(iq) < 100:
        return SNRResult(None, "Spectral", None, None, False, "Insufficient samples")
        
    from rf_analyzer.spectral.features import power_spectral_density, detect_occupied_band
    
    iq = np.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0)
    freqs, psd = power_spectral_density(iq, sample_rate)
    
    if occupied_band is None:
        band_info = detect_occupied_band(freqs, psd)
        if band_info is None:
            return SNRResult(None, "Spectral", None, None, False, "Could not detect occupied band")
        f_low, f_high = band_info["f_lo"], band_info["f_hi"]
    else:
        f_low, f_high = occupied_band
        
    in_band_mask = (freqs >= f_low) & (freqs <= f_high)
    out_band_mask = ~in_band_mask
    
    if not np.any(in_band_mask):
        return SNRResult(None, "Spectral", None, None, False, "Occupied band has no spectral bins")
        
    if not np.any(out_band_mask):
        return SNRResult(None, "Spectral", None, None, False, "Signal occupies full band; no guard band for noise estimation")
        
    mean_psd_in = np.mean(psd[in_band_mask])
    mean_psd_out = np.mean(psd[out_band_mask])
    
    if mean_psd_out <= 0:
        mean_psd_out = 1e-15
        
    signal_psd = mean_psd_in - mean_psd_out
    
    if signal_psd <= 0:
        return SNRResult(None, "Spectral", None, None, False, "Negative signal power estimate (noise floor >= signal band power)")
        
    snr_linear = signal_psd / mean_psd_out
    snr_db = 10 * np.log10(snr_linear)
    
    num_in_bins = np.sum(in_band_mask)
    df = freqs[1] - freqs[0]
    total_signal_power = signal_psd * num_in_bins * df
    total_noise_power_in_band = mean_psd_out * num_in_bins * df
    
    return SNRResult(float(snr_db), "Spectral", float(total_signal_power), float(total_noise_power_in_band), True)
