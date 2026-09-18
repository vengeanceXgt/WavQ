"""Stage 9: Demodulation to soft-decision LLRs."""
import numpy as np
from rf_analyzer.utils.synth import CONSTELLATIONS

def symbols_to_llr(symbols, scheme, noise_var=0.1):
    """LLR > 0 means 'more likely a 0 bit', LLR < 0 means 'more likely a 1 bit'
    (matches the convention used by the FEC decoder in Phase 8)."""
    if scheme == "2fsk":
        inst_freq = np.angle(symbols[1:] * np.conj(symbols[:-1]))
        return (-inst_freq / max(noise_var, 1e-6)).astype(np.float32)

    const = CONSTELLATIONS[scheme]
    bits_per_sym = int(np.log2(len(const)))
    llrs = np.empty(len(symbols) * bits_per_sym, dtype=np.float32)
    for n, s in enumerate(symbols):
        dists = np.abs(s - const) ** 2
        for b in range(bits_per_sym):
            zero_mask = ((np.arange(len(const)) >> (bits_per_sym - 1 - b)) & 1) == 0
            d0 = dists[zero_mask].min()
            d1 = dists[~zero_mask].min()
            llrs[n * bits_per_sym + b] = (d1 - d0) / max(noise_var, 1e-6)
    return llrs

def llr_to_hard_bits(llr):
    return (llr < 0).astype(np.uint8)
