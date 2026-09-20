import numpy as np
from rf_analyzer.fec.viterbi import encode_conv, viterbi_decode
from rf_analyzer.fec.identify import identify_and_decode
from rf_analyzer.utils.synth import random_bits

def _encode_to_llr(bits, noise_scale=0.7, seed=0):
    coded = encode_conv(bits)
    rng = np.random.default_rng(seed)
    llr = np.where(coded == 1, -3.0, 3.0).astype(np.float32)
    llr += rng.normal(scale=noise_scale, size=llr.shape)
    return llr

def test_FEC01_viterbi_corrects_moderate_noise():
    bits = random_bits(300, seed=5)
    llr = _encode_to_llr(bits, noise_scale=0.7)
    decoded, _ = viterbi_decode(llr)
    n = min(len(bits), len(decoded))
    ber = np.mean(bits[:n] != decoded[:n])
    assert ber < 0.01

def test_FEC02_identification_positive_case():
    bits = random_bits(300, seed=6)
    llr = _encode_to_llr(bits, noise_scale=0.7)
    result = identify_and_decode(llr)
    assert result["fec_type"] == "conv_r1_2_k7"
    assert result["validation_evidence"]["evidence_score"] > 0.7

def test_FEC03_identification_negative_case_no_fec():
    bits = random_bits(300, seed=7)
    llr = np.where(bits == 1, -3.0, 3.0).astype(np.float32)
    rng = np.random.default_rng(8)
    llr += rng.normal(scale=0.7, size=llr.shape)
    result = identify_and_decode(llr)
    assert result["fec_type"] == "unknown"

def test_FEC04_waterfall_threshold_documented():
    bits = random_bits(500, seed=9)
    for noise_scale in [0.3, 0.6, 0.9, 1.2, 1.5]:
        llr = _encode_to_llr(bits, noise_scale=noise_scale, seed=1)
        decoded, _ = viterbi_decode(llr)
        n = min(len(bits), len(decoded))
        errors = int(np.sum(bits[:n] != decoded[:n]))
        print(f"noise_scale={noise_scale}: errors={errors}")
