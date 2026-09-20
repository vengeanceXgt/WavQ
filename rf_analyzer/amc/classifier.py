"""Stage 7: Automatic Modulation Classification (MVP) using Exact Mixture Likelihood."""
import numpy as np
from rf_analyzer.amc.exact_ml import estimate_nuisance_parameters, calc_exact_log_likelihood

def _moment(iq, p, q):
    return np.mean((iq ** (p - q)) * (np.conj(iq) ** q))

def cumulant_features(iq):
    iq = iq / np.sqrt(np.mean(np.abs(iq) ** 2) + 1e-15)
    m20 = _moment(iq, 2, 0)
    m21 = _moment(iq, 2, 1)
    m40 = _moment(iq, 4, 0)
    m42 = _moment(iq, 4, 2)
    c40 = m40 - 3 * m20 ** 2
    c42 = m42 - np.abs(m20) ** 2 - 2 * m21 ** 2
    return {"c40": c40, "c42": c42, "|c40|": abs(c40), "|c42|": abs(c42)}

def is_constant_envelope(continuous_iq, tolerance=0.15):
    mag = np.abs(continuous_iq)
    return float(np.std(mag) / (np.mean(mag) + 1e-15)) < tolerance

_REFERENCE = {
    "bpsk":  {"c40": 1.49, "c42": 1.57},
    "qpsk":  {"c40": 0.76, "c42": 1.00},
    "16qam": {"c40": 0.61, "c42": 0.69},
}

def get_constellations():
    qam16 = np.array([-3, -1, 1, 3])
    x, y = np.meshgrid(qam16, qam16)
    return {
        "bpsk": np.array([1, -1]) + 0j,
        "qpsk": np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2),
        "16qam": (x.flatten() + 1j*y.flatten()) / np.sqrt(10)
    }

def classify_modulation(continuous_iq, decimated_symbols=None, snr_db=None):
    """
    Classify modulation using Exact ML + Cumulants.
    """
    if is_constant_envelope(continuous_iq):
        return {
            "label": "2fsk",
            "evidence": {"method": "envelope", "reason": "constant envelope"},
            "ambiguous": False,
            "candidates": [{"label": "2fsk", "score": 1.0}]
        }
        
    symbols = decimated_symbols if decimated_symbols is not None else continuous_iq
    
    # Optional Cumulant Fusion if SNR is available (default 10dB)
    snr = snr_db if snr_db is not None else 10.0
    snr_lin = 10**(snr/10)
    rho_sq = (snr_lin / (snr_lin + 1))**2
    
    f = cumulant_features(symbols)
    c40 = f["|c40|"] / (rho_sq + 1e-9)
    c42 = f["|c42|"] / (rho_sq + 1e-9)
    
    consts = get_constellations()
    scores = {}
    details = []
    
    for hyp, const in consts.items():
        st, A, p, sig = estimate_nuisance_parameters(symbols, const)
        ll = calc_exact_log_likelihood(st, const, sig)
        
        # Cumulant distance penalty
        ref = _REFERENCE[hyp]
        dist = (c40 - ref["c40"])**2 + (c42 - ref["c42"])**2
        combo_ll = ll - 50.0 * dist
        
        scores[hyp] = float(combo_ll)
        details.append({"label": hyp, "score": float(combo_ll), "sigma_sq": float(sig)})
        
    sorted_hyps = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    best = sorted_hyps[0]
    second = sorted_hyps[1]
    margin = scores[best] - scores[second]
    ambiguous = margin < 10.0
    
    return {
        "label": best,
        "evidence": {
            "method": "exact_ml_cumulants",
            "margin": margin,
            "best_score": scores[best],
            "second_score": scores[second],
            "cumulants": f
        },
        "ambiguous": ambiguous,
        "candidates": sorted(details, key=lambda x: x["score"], reverse=True)
    }

def classify_modulation_auto(continuous_iq, decimated_symbols=None, raw_iq=None, snr_db=None):
    # MVP limits us to the proven exact ML classifier for FSK/PSK/QAM.
    return classify_modulation(continuous_iq, decimated_symbols, snr_db=snr_db)
