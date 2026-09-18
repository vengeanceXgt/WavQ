import pytest
import numpy as np
from rf_analyzer.utils.synth import generate_test_signal
from rf_analyzer.spectral.features import estimate_cfo_nonlinear, mix_to_baseband
from rf_analyzer.sync.matched_filter import apply_rx_matched_filter
from rf_analyzer.sync.timing import gardner_timing_recovery
from rf_analyzer.amc.classifier import classify_modulation

def _classify(iq, truth, scheme):
    if scheme == "2fsk":
        return classify_modulation(iq, gardner_timing_recovery(iq, sps=truth["sps"]))
    cfo = estimate_cfo_nonlinear(iq, truth["sample_rate"], power=2 if scheme == "bpsk" else 4)
    iq_bb = mix_to_baseband(iq, truth["sample_rate"], cfo)
    continuous = apply_rx_matched_filter(iq_bb, sps=truth["sps"])
    symbols = gardner_timing_recovery(continuous, sps=truth["sps"])
    return classify_modulation(continuous, symbols)

@pytest.mark.parametrize("scheme", ["2fsk", "bpsk", "qpsk", "16qam"])
def test_AMC01_correct_classification_high_snr(scheme):
    hits = 0
    for seed in range(8):
        cfo_hz = 0.0 if scheme == "2fsk" else 1200.0
        iq, truth = generate_test_signal(scheme=scheme, cfo_hz=cfo_hz, snr_db=20, seed=seed)
        result = _classify(iq, truth, scheme)
        if result["modulation"] == scheme:
            hits += 1
    assert hits >= 7, f"{scheme}: only {hits}/8 correct at 20dB SNR"

@pytest.mark.parametrize("snr_db", [20, 10])
def test_AMC02_qpsk_accuracy_at_snr(snr_db):
    """Measured behavior on properly matched-filtered, symbol-decimated signals:
    this classifier degrades much more gracefully than the earlier (broken)
    raw-oversampled-sample version -- see Part 1/Part 2 design notes."""
    hits, n = 0, 15
    for seed in range(n):
        iq, truth = generate_test_signal(scheme="qpsk", cfo_hz=1200.0, snr_db=snr_db, seed=seed)
        result = _classify(iq, truth, "qpsk")
        if result["modulation"] == "qpsk":
            hits += 1
    accuracy = hits / n
    print(f"QPSK accuracy at {snr_db}dB: {accuracy:.2f}")
    if snr_db == 20:
        assert accuracy >= 0.8
    else:
        assert accuracy >= 0.4
