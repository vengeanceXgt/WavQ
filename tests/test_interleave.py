import numpy as np
from rf_analyzer.interleave.block import detect_block_interleaver_width, deinterleave_block, gf2_rank

def _make_block_interleaved(width=11, rows=16, seed=0, n_independent=6):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 2, size=(n_independent, width), dtype=np.uint8)
    matrix = np.zeros((rows, width), dtype=np.uint8)
    matrix[:n_independent] = base
    for r in range(n_independent, rows):
        i, j = rng.integers(0, n_independent, size=2)
        matrix[r] = base[i] ^ base[j]
    interleaved = matrix.T.reshape(-1)
    return interleaved, width, rows, matrix

def test_INT01_detects_known_block_width():
    hits = 0
    for seed in range(5):
        bits, true_width, rows, _ = _make_block_interleaved(seed=seed)
        result = detect_block_interleaver_width(bits, min_width=4, max_width=24, rows=rows)
        if result and result["width"] == true_width:
            hits += 1
    assert hits >= 4

def test_INT02_negative_case_random_bits_no_false_positive():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=16*24, dtype=np.uint8)
    result = detect_block_interleaver_width(bits, min_width=4, max_width=24, rows=16)
    assert result is not None
    assert result["confidence"] < 0.3

def test_INT03_deinterleave_recovers_original_order():
    bits, width, rows, orig_matrix = _make_block_interleaved(seed=2)
    recovered = deinterleave_block(bits, width, rows)
    assert np.array_equal(recovered, orig_matrix.reshape(-1))

def test_INT04_gf2_rank_known_values():
    identity = np.eye(4, dtype=np.uint8)
    assert gf2_rank(identity) == 4
    all_same_row = np.tile(np.array([1,0,1,0], dtype=np.uint8), (4,1))
    assert gf2_rank(all_same_row) == 1
