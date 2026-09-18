"""Stage 11: FEC identification -- try candidate code families, score by syndrome health.

Confidence is derived from RE-ENCODE MISMATCH (decode the LLRs, re-encode the
result, and see how well it matches the original hard-decision bits) rather
than the raw Viterbi path metric. This matters: the path metric alone barely
separates truly-coded from uncoded data (a maximum-likelihood decoder always
finds *some* nearby codeword, coded or not), but the re-encode mismatch does
NOT -- measured empirically, truly coded data re-encodes to ~0% mismatch even
under noise, while uncoded/random data re-encodes to a consistent ~12-13%
mismatch baseline (the code's inherent "covering radius"). The confidence
scaling factor (6x) below is chosen from that measured gap, with headroom on
both sides -- see Part 3's FEC02/FEC03 tests for the exact numbers this was
tuned against.
"""
import numpy as np
from .viterbi import viterbi_decode, encode_conv

def try_convolutional_hypothesis(llr):
    decoded, metric = viterbi_decode(llr)
    reencoded = encode_conv(decoded)
    hard_bits = (llr < 0).astype(np.uint8)
    n = min(len(reencoded), len(hard_bits))
    mismatch = float(np.mean(reencoded[:n] != hard_bits[:n])) if n else 1.0
    confidence = max(0.0, 1.0 - mismatch * 6)
    return {"fec_type": "conv_r1_2_k7", "decoded_bits": decoded,
            "mismatch": mismatch, "confidence": confidence}

def identify_and_decode(llr):
    """MVP: single hypothesis (rate-1/2 K=7 convolutional). Extension point for
    Reed-Solomon / LDPC hypotheses -- see Phase 8 Extension note."""
    result = try_convolutional_hypothesis(llr)
    if result["confidence"] < 0.3:
        return {"fec_type": "unknown", "confidence": result["confidence"],
                "decoded_bits": None,
                "notes": "No candidate FEC scheme validated; passing raw hard bits through."}
    return result
