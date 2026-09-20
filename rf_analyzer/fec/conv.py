import numpy as np
from .base import FECScheme, FECResult
from .viterbi import viterbi_decode, encode_conv

class ConvFEC(FECScheme):
    @property
    def name(self) -> str:
        return "conv_r1_2_k7"

    def identify_and_decode(self, llr) -> FECResult:
        """
        Evidence score is derived from RE-ENCODE MISMATCH (decode the LLRs, re-encode the
        result, and see how well it matches the original hard-decision bits). truly coded
        data re-encodes to ~0% mismatch even under noise, while uncoded/random data
        re-encodes to a consistent ~12-13% mismatch baseline (the code's inherent
        "covering radius"). The evidence scaling factor (6x) below is chosen from that
        measured gap, with headroom on both sides.
        """
        decoded, metric = viterbi_decode(llr)
        reencoded = encode_conv(decoded)
        hard_bits = (llr < 0).astype(np.uint8)
        n = min(len(reencoded), len(hard_bits))
        mismatch = float(np.mean(reencoded[:n] != hard_bits[:n])) if n else 1.0
        evidence_score = max(0.0, 1.0 - mismatch * 6)
        
        return FECResult(
            fec_type=self.name,
            decoded_bits=decoded,
            validation_evidence={"evidence_score": evidence_score, "reencode_mismatch_rate": mismatch}
        )
