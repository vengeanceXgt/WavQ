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
