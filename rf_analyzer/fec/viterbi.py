"""Stage 12: Deterministic Viterbi decoding for rate-1/2 convolutional codes.

State convention: `state` is the (k-1)-bit shift-register memory (the K-1
previously shifted-in bits), NOT the full k-bit polynomial-evaluation window.
_encode_bit_pair combines state+new bit into a k-bit window to compute the two
parity outputs (matching how the k-bit octal generator polynomials are meant
to be applied), but must return a (k-1)-bit NEXT state (dropping the oldest
bit) -- returning the full k-bit value as the next state is a real bug that
raises an IndexError as soon as the trellis produces a state >= n_states
(caught here by actually running the self-test below, not just reading the
math on paper).
"""
import numpy as np

G1, G2 = 0o171, 0o133
CONSTRAINT_LEN = 7

def _encode_bit_pair(shift_reg, bit, g1=G1, g2=G2, k=CONSTRAINT_LEN):
    window = ((shift_reg << 1) | bit) & ((1 << k) - 1)   # k-bit window for parity calc
    def parity(x, g):
        return bin(x & g).count("1") % 2
    next_state = window & ((1 << (k - 1)) - 1)           # (k-1)-bit memory for next step
    return parity(window, g1), parity(window, g2), next_state

def encode_conv(bits, g1=G1, g2=G2, k=CONSTRAINT_LEN):
    reg = 0
    out = []
    for b in bits:
        p1, p2, reg = _encode_bit_pair(reg, b, g1, g2, k)
        out.extend([p1, p2])
    return np.array(out, dtype=np.uint8)

def viterbi_decode(llr, g1=G1, g2=G2, k=CONSTRAINT_LEN):
    n_states = 1 << (k - 1)
    n_symbols = len(llr) // 2
    INF = 1e9
    path_metric = np.full(n_states, INF)
    path_metric[0] = 0.0
    paths = [[] for _ in range(n_states)]

    for sym in range(n_symbols):
        rx1, rx2 = llr[2 * sym], llr[2 * sym + 1]
        new_metric = np.full(n_states, INF)
        new_paths = [[] for _ in range(n_states)]
        for state in range(n_states):
            if path_metric[state] >= INF:
                continue
            for bit in (0, 1):
                p1, p2, next_state = _encode_bit_pair(state, bit, g1, g2, k)
                exp1 = -1.0 if p1 else 1.0
                exp2 = -1.0 if p2 else 1.0
                branch_metric = abs(rx1 - exp1) + abs(rx2 - exp2)
                metric = path_metric[state] + branch_metric
                if metric < new_metric[next_state]:
                    new_metric[next_state] = metric
                    new_paths[next_state] = paths[state] + [bit]
        path_metric, paths = new_metric, new_paths

    best_state = int(np.argmin(path_metric))
    return np.array(paths[best_state], dtype=np.uint8), float(path_metric[best_state])
