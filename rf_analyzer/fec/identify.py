"""Stage 11: FEC identification -- try candidate code families, score by syndrome health.
"""
from .conv import ConvFEC
from .rs import ReedSolomonFEC

# Registry of supported FEC schemes
_SCHEMES = [
    ConvFEC(),
    ReedSolomonFEC(255, 223),
]

def identify_and_decode(llr):
    """
    Tries candidate FEC schemes and returns the one with the highest evidence score.
    Extension point for Reed-Solomon / LDPC hypotheses -- see Phase 8 Extension note.
    """
    best_result = None
    best_score = -1.0
    
    for scheme in _SCHEMES:
        res = scheme.identify_and_decode(llr)
        if res is not None:
            score = res.validation_evidence.get("evidence_score", 0.0)
            if score > best_score:
                best_score = score
                best_result = res
                
    if best_result is not None and best_score >= 0.3:
        return best_result.to_dict()
        
    return {
        "fec_type": "unknown", 
        "validation_evidence": best_result.validation_evidence if best_result else None,
        "decoded_bits": None,
        "notes": "No candidate FEC scheme validated; passing raw hard bits through."
    }
