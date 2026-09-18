"""Stage 6: Symbol-Rate Estimation via a transition-detector spectrum.

Uses the instantaneous-frequency difference (phase-jump detector) rather than
a plain magnitude-squared spectrum, because magnitude-squared carries NO timing
information for constant-modulus modulations (PSK/QAM) sent with rectangular
(unshaped) pulses -- their envelope is perfectly flat. The instantaneous-frequency
difference instead detects the phase JUMPS at symbol boundaries, which exist
for BPSK/QPSK/QAM regardless of pulse shaping. (Continuous-phase FSK is a known
harder case -- see the note in the Debugging Plan.)
"""
import numpy as np

def estimate_symbol_rate(iq, sample_rate, min_sps=2, max_sps=32):
    inst_freq = np.angle(iq[1:] * np.conj(iq[:-1]))
    transitions = np.abs(np.diff(inst_freq))
    transitions = transitions - np.mean(transitions)
    spectrum = np.abs(np.fft.rfft(transitions))
    freqs = np.fft.rfftfreq(len(transitions), d=1.0 / sample_rate)
    valid = (freqs > sample_rate / (max_sps * 4)) & (freqs > 1.0)
    if not np.any(valid):
        return None
    idx = np.argmax(spectrum * valid)
    symbol_rate_hz = float(freqs[idx])
    sps = sample_rate / symbol_rate_hz if symbol_rate_hz > 0 else None
    return {"symbol_rate_hz": symbol_rate_hz, "samples_per_symbol": sps}
