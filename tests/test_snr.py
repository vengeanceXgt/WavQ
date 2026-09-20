import numpy as np
from rf_analyzer.eval.snr import estimate_snr_m2m4, estimate_snr_spectral
from rf_analyzer.utils.synth import generate_test_signal

def _generate_qpsk(length, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    symbols = rng.choice([1+1j, 1-1j, -1+1j, -1-1j], size=length) / np.sqrt(2)
    noise_power = 10**(-snr_db/10)
    noise = (rng.normal(scale=np.sqrt(noise_power/2), size=length) + 
             1j * rng.normal(scale=np.sqrt(noise_power/2), size=length))
    return symbols + noise

def test_snr_m2m4_accuracy():
    symbols = _generate_qpsk(10000, 15.0)
    result = estimate_snr_m2m4(symbols)
    assert result.valid
    assert abs(result.snr_db - 15.0) < 1.0

def test_snr_spectral_accuracy():
    iq, truth = generate_test_signal(scheme="qpsk", sps=8, snr_db=12.0, sample_rate=1e6, seed=1)
    result = estimate_snr_spectral(iq, sample_rate=1e6)
    assert result.valid
    # Spectral estimator calculates in-band SNR.
    # The synthetic generator adds noise across the full Nyquist band.
    # Since SPS=8, the signal occupies ~1/8 of the band, making in-band SNR ~9dB higher.
    # Expected in-band SNR ≈ 12.0 + 10*log10(8) ≈ 21.0 dB
    assert abs(result.snr_db - 21.0) < 3.5

def test_snr_spectral_no_noise_band():
    rng = np.random.default_rng(0)
    iq = (rng.normal(scale=1.0, size=1000) + 1j * rng.normal(scale=1.0, size=1000)).astype(np.complex64)
    # Give it the full band so there are no out-of-band bins
    result = estimate_snr_spectral(iq, sample_rate=1e6, occupied_band=(-5e5, 5e5))
    assert not result.valid
    assert "no guard band" in result.reason
    
def test_snr_m2m4_insufficient_samples():
    symbols = _generate_qpsk(10, 15.0)
    result = estimate_snr_m2m4(symbols)
    assert not result.valid
    assert "Insufficient" in result.reason
