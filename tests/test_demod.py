import numpy as np
from rf_analyzer.utils.synth import generate_test_signal
from rf_analyzer.sync.matched_filter import apply_rx_matched_filter
from rf_analyzer.sync.timing import gardner_timing_recovery
from rf_analyzer.demod.llr import symbols_to_llr, llr_to_hard_bits

def test_DEMOD01_low_ber_at_high_snr_no_cfo():
    for scheme in ["bpsk", "qpsk", "16qam"]:
        iq, truth = generate_test_signal(scheme=scheme, sps=8, snr_db=25, cfo_hz=0.0)
        filtered = apply_rx_matched_filter(iq, sps=8)
        symbols = gardner_timing_recovery(filtered, sps=8)
        llr = symbols_to_llr(symbols, scheme)
        bits = llr_to_hard_bits(llr)
        n = min(len(bits), len(truth["bits"]))
        ber = np.mean(bits[:n] != truth["bits"][:n])
        assert ber < 0.05, f"{scheme}: BER={ber:.3f} too high at 25dB"

def test_DEMOD02_llr_sign_convention_consistency():
    iq, truth = generate_test_signal(scheme="bpsk", sps=8, snr_db=25, cfo_hz=0.0)
    filtered = apply_rx_matched_filter(iq, sps=8)
    symbols = gardner_timing_recovery(filtered, sps=8)
    llr = symbols_to_llr(symbols, "bpsk")
    bits_correct = llr_to_hard_bits(llr)
    bits_flipped = llr_to_hard_bits(-llr)
    n = min(len(bits_correct), len(truth["bits"]))
    ber_correct = np.mean(bits_correct[:n] != truth["bits"][:n])
    ber_flipped = np.mean(bits_flipped[:n] != truth["bits"][:n])
    assert ber_correct < 0.05
    assert ber_flipped > 0.9
