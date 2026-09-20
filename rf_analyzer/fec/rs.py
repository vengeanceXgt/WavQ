import numpy as np
import reedsolo
from .base import FECScheme, FECResult

class ReedSolomonFEC(FECScheme):
    """
    Reed-Solomon FEC decoder hypothesis using the reedsolo library.
    Defaults to CCSDS standard RS(255, 223).
    """
    def __init__(self, n_bytes=255, k_bytes=223):
        self.n = n_bytes
        self.k = k_bytes
        self.rs = reedsolo.RSCodec(n_bytes - k_bytes)
        
    @property
    def name(self) -> str:
        return f"rs_{self.n}_{self.k}"

    def identify_and_decode(self, llr):
        # Convert LLR to hard bits (0 or 1). LLR < 0 means bit 1 (assuming BPSK mapping +1->0, -1->1)
        hard_bits = (llr < 0).astype(np.uint8)
        
        block_size_bits = self.n * 8
        if len(hard_bits) < block_size_bits:
            return None
            
        n_blocks = len(hard_bits) // block_size_bits
        bits_to_use = hard_bits[:n_blocks * block_size_bits]
        
        # Pack bits into bytes (MSB first)
        packed_bytes = np.packbits(bits_to_use)
        
        valid_blocks = 0
        decoded_bytes = bytearray()
        
        for i in range(n_blocks):
            block = packed_bytes[i * self.n : (i + 1) * self.n]
            
            try:
                dec, _, _ = self.rs.decode(block)
                decoded_bytes.extend(dec)
                valid_blocks += 1
            except reedsolo.ReedSolomonError:
                # Decoding failed for this block
                pass
                
        if valid_blocks == 0:
            return None
            
        decoded_array = np.frombuffer(decoded_bytes, dtype=np.uint8)
        decoded_bits = np.unpackbits(decoded_array)
        
        # Evidence score is the ratio of valid decoded blocks
        score = valid_blocks / n_blocks
        
        return FECResult(
            fec_type=self.name,
            decoded_bits=decoded_bits,
            validation_evidence={"evidence_score": score, "valid_blocks": valid_blocks, "total_blocks": n_blocks}
        )
