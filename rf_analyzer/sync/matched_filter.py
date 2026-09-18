"""Stage 8 (part 0, previously MISSING): receive matched filtering.

Any real digital-communications transmitter applies pulse shaping (typically
RRC) before transmission, precisely because it wastes far less bandwidth than
unshaped pulses. A real receiver correspondingly applies a matched filter
(the same RRC response) before symbol timing recovery -- this maximizes SNR
at the sampling instant and is a standard, necessary receiver stage. The
original MVP pipeline skipped this stage entirely (it went straight from
carrier mixing to Gardner timing recovery), which was never caught because
the original synthetic test signals had no pulse shaping to match-filter
against in the first place. This stage is now part of the pipeline -- see
Part 1's Phase 5 update in the implementation plan.
"""
import numpy as np
from rf_analyzer.utils.synth import rrc_filter

def apply_rx_matched_filter(iq, rrc_beta=0.35, rrc_span=6, sps=8):
    h = rrc_filter(rrc_beta, rrc_span, sps)
    filtered = np.convolve(iq, h, mode="same")
    return filtered.astype(np.complex64)
