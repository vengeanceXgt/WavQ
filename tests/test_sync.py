import numpy as np
from rf_analyzer.utils.synth import generate_test_signal
from rf_analyzer.spectral.features import estimate_cfo_nonlinear, mix_to_baseband
from rf_analyzer.sync.matched_filter import apply_rx_matched_filter
from rf_analyzer.sync.timing import gardner_timing_recovery
from rf_analyzer.sync.carrier import costas_loop_qpsk

def test_SYNC01_symbol_count_matches_expected():
    iq, truth = generate_test_signal(scheme="qpsk", sps=8, snr_db=20, cfo_hz=0.0)
    filtered = apply_rx_matched_filter(iq, sps=8)
    symbols = gardner_timing_recovery(filtered, sps=8)
    expected = len(iq) // 8
    assert abs(len(symbols) - expected) / expected < 0.1

def test_SYNC02_costas_loop_reduces_phase_rotation():
    iq, truth = generate_test_signal(scheme="qpsk", sps=8, snr_db=25, cfo_hz=300.0)
    cfo = estimate_cfo_nonlinear(iq, truth["sample_rate"], power=4)
    iq_bb = mix_to_baseband(iq, truth["sample_rate"], cfo)
    filtered = apply_rx_matched_filter(iq_bb, sps=8)
    symbols = gardner_timing_recovery(filtered, sps=8)
    locked = costas_loop_qpsk(symbols)
    tail = locked[len(locked)//2:]
    residual_phase = np.angle(tail) % (np.pi / 2)
    assert np.std(residual_phase) < 0.5

def test_SYNC03_matched_filter_required_for_low_ber():
    """Regression guard: confirms the matched-filter stage is not accidentally
    skipped -- without it, BER should be visibly worse even at high SNR."""
    from rf_analyzer.demod.llr import symbols_to_llr, llr_to_hard_bits
    iq, truth = generate_test_signal(scheme="16qam", sps=8, snr_db=25, cfo_hz=0.0)
    filtered = apply_rx_matched_filter(iq, sps=8)
    with_mf = gardner_timing_recovery(filtered, sps=8)
    without_mf = gardner_timing_recovery(iq, sps=8)
    llr_with = symbols_to_llr(with_mf, "16qam")
    llr_without = symbols_to_llr(without_mf, "16qam")
    ber_with = np.mean(llr_to_hard_bits(llr_with)[:len(truth["bits"])] != truth["bits"][:len(llr_to_hard_bits(llr_with))])
    n = min(len(llr_to_hard_bits(llr_without)), len(truth["bits"]))
    ber_without = np.mean(llr_to_hard_bits(llr_without)[:n] != truth["bits"][:n])
    assert ber_with < 0.05
    assert ber_without > ber_with
