import numpy as np
from rf_analyzer.utils.synth import generate_test_signal
from rf_analyzer.spectral.features import (power_spectral_density, detect_occupied_band,
                                            estimate_cfo_nonlinear, mix_to_baseband)
from rf_analyzer.spectral.symbol_rate import estimate_symbol_rate

def test_SPEC01_cfo_estimation_within_tolerance():
    hits = 0
    for seed in range(5):
        iq, truth = generate_test_signal(scheme="qpsk", cfo_hz=1500.0, snr_db=20, seed=seed)
        est = estimate_cfo_nonlinear(iq, truth["sample_rate"], power=4)
        if abs(est - truth["cfo_hz"]) < 200:
            hits += 1
    assert hits >= 4

def test_SPEC02_band_detection_finds_signal():
    iq, truth = generate_test_signal(snr_db=20)
    freqs, psd = power_spectral_density(iq, truth["sample_rate"])
    band = detect_occupied_band(freqs, psd)
    assert band is not None
    assert band["bandwidth_hz"] > 0

def test_SPEC03_symbol_rate_estimation():
    hits = 0
    for seed in range(5):
        iq, truth = generate_test_signal(sps=8, sample_rate=1e6, snr_db=20, seed=seed)
        est = estimate_symbol_rate(iq, truth["sample_rate"])
        if est and abs(est["samples_per_symbol"] - 8) <= 1:
            hits += 1
    assert hits >= 4

def test_SPEC04_edge_case_no_signal_present():
    rng = np.random.default_rng(0)
    noise = (rng.normal(size=50000) + 1j * rng.normal(size=50000)).astype(np.complex64)
    freqs, psd = power_spectral_density(noise, 1e6)
    band = detect_occupied_band(freqs, psd, threshold_db_above_floor=6.0)
    assert band is None
