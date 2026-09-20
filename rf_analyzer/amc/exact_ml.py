import numpy as np
from scipy.special import logsumexp

def estimate_nuisance_parameters(symbols, const):
    """
    Given an observation block of symbols, estimates:
    A (amplitude), theta (phase), and sigma_sq (variance).
    Uses a standard nearest-neighbor blind phase search.
    """
    rms_sig = np.sqrt(np.mean(np.abs(symbols)**2))
    rms_const = np.sqrt(np.mean(np.abs(const)**2))
    A = rms_sig / (rms_const + 1e-15)
    st = symbols / A
    
    best_phase = 0.0
    min_err = float('inf')
    
    for p in np.linspace(0, 2*np.pi, 32, endpoint=False):
        rotated = st * np.exp(-1j * p)
        diffs = rotated[:, None] - const[None, :]
        min_dist_sq = np.min(np.abs(diffs)**2, axis=1)
        err = np.mean(min_dist_sq)
        if err < min_err:
            min_err = err
            best_phase = p
            
    st_rot = st * np.exp(-1j * best_phase)
    diffs = st_rot[:, None] - const[None, :]
    min_dist_sq = np.min(np.abs(diffs)**2, axis=1)
    sigma_sq = np.mean(min_dist_sq) + 1e-15
    
    return st_rot, A, best_phase, sigma_sq

def calc_exact_log_likelihood(symbols, const, sigma_sq):
    """
    Computes exact logsumexp mixture likelihood for the constellation.
    """
    M = len(const)
    diffs = symbols[:, None] - const[None, :]
    dist_sq = np.abs(diffs)**2
    exponents = -dist_sq / sigma_sq
    
    log_sum_terms = logsumexp(exponents, axis=1)
    ll_per_sym = log_sum_terms - np.log(M) - np.log(np.pi * sigma_sq)
    return np.sum(ll_per_sym)
