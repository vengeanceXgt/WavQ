import numpy as np
import reedsolo
from rf_analyzer.fec.rs import ReedSolomonFEC

def test_RS_identifies_valid_frame():
    # Create RS(255, 223)
    rs_codec = reedsolo.RSCodec(32)
    
    # 223 bytes of random data
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=223, dtype=np.uint8)
    
    # Encode to 255 bytes
    encoded = rs_codec.encode(bytearray(data))
    
    # Convert to bits
    bits = np.unpackbits(np.array(encoded, dtype=np.uint8))
    
    # BPSK modulate: 0 -> +3.0, 1 -> -3.0
    llr = np.where(bits == 1, -3.0, 3.0).astype(np.float32)
    
    # Add a few errors (e.g. flip 5 bits)
    # 5 bit errors is well within the 16 byte error correction capability
    error_indices = rng.choice(len(llr), size=5, replace=False)
    llr[error_indices] *= -1
    
    # Test identification
    scheme = ReedSolomonFEC(255, 223)
    result = scheme.identify_and_decode(llr)
    
    assert result is not None
    assert result.fec_type == "rs_255_223"
    assert result.validation_evidence["evidence_score"] == 1.0
    assert result.validation_evidence["valid_blocks"] == 1
    
    # Verify decoded bits match original data
    original_bits = np.unpackbits(data)
    assert np.array_equal(result.decoded_bits, original_bits)

def test_RS_rejects_noise():
    # Random LLRs representing noise
    rng = np.random.default_rng(99)
    llr = rng.normal(scale=3.0, size=255 * 8 * 2) # 2 blocks
    
    scheme = ReedSolomonFEC(255, 223)
    result = scheme.identify_and_decode(llr)
    
    # Should completely fail to decode both blocks
    assert result is None
