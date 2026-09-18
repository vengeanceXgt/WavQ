"""Stage 10: Blind block-interleaver identification via GF(2) rank deficiency.

Interleaving convention used throughout this module: the original payload is
written row-major into a (rows x width) matrix, then TRANSMITTED column-major
(matrix.T.reshape(-1)) -- the standard block-interleave definition. That means
the transmitted stream's natural shape is (width, rows), not (rows, width).
An earlier version of this code reshaped candidate chunks as (rows, width)
directly, which silently tested the WRONG matrix orientation and could never
reliably find the true width (confirmed by testing against synthetic data with
a known, deliberately-injected linear dependency at a known width -- the
"wrong-orientation" version detected an unrelated width on every run). Always
reshape as (width, rows) to match the transmission order, matching
deinterleave_block below.
"""
import numpy as np

def gf2_rank(matrix):
    """Rank of a binary matrix over GF(2) via Gaussian elimination with XOR."""
    m = matrix.copy().astype(np.uint8)
    rows, cols = m.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if m[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(rows):
            if r != rank and m[r, col] == 1:
                m[r] ^= m[rank]
        rank += 1
        if rank == rows:
            break
    return rank

def detect_block_interleaver_width(bits, min_width=4, max_width=64, rows=8):
    """Reshapes the bitstream into candidate matrices of varying width and looks
    for a width where the rank is *deficient* relative to full rank -- interleaved
    (structured) data produces linear dependencies that random data doesn't."""
    results = []
    for width in range(min_width, max_width + 1):
        needed = width * rows
        if needed > len(bits):
            break
        chunk = bits[:needed].reshape(width, rows)   # (width, rows): matches transmission order
        rank = gf2_rank(chunk)
        deficiency = min(rows, width) - rank
        results.append({"width": width, "rank": rank, "deficiency": deficiency})
    if not results:
        return None
    best = max(results, key=lambda r: r["deficiency"])
    confidence = best["deficiency"] / min(rows, best["width"])
    return {"width": best["width"], "confidence": float(confidence), "all_results": results}

def deinterleave_block(bits, width, rows):
    n = width * rows
    matrix = bits[:n].reshape(width, rows)  # matches the transmission shape
    return matrix.T.reshape(-1)             # transpose back to (rows, width) row-major order
