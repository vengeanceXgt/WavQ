"""Stage 7: Automatic Modulation Classification (MVP).

Two-stage design, in this specific order for a specific reason (found by
testing against realistically pulse-shaped signals, not the idealized
rectangular pulses used in earlier iterations of this module):

  1. Envelope-constancy check on the continuous (matched-filtered, NOT YET
     symbol-decimated) baseband waveform separates FSK from the PSK/QAM
     family. An earlier version tried this on UNSHAPED rectangular-pulse
     signals and found BPSK/QPSK were ALSO constant-envelope there, so this
     check was removed. That turned out to be the wrong lesson: unshaped PSK
     is an unrealistic idealization. Realistically RRC-shaped PSK/QAM has
     genuine envelope variation (from inter-symbol pulse overlap), while
     FSK's constant-modulus property survives pulse shaping -- so this check
     IS valid, as long as it's measured on a properly shaped signal.

  2. Higher-order-cumulant nearest-neighbor match separates BPSK/QPSK/16-QAM,
     computed on the DECIMATED, SYMBOL-RATE samples coming out of timing
     recovery (Stage 8) -- NOT on raw oversampled IQ. Oversampled samples
     carry artificial sample-to-sample correlation from pulse shaping that
     distorts the cumulant statistics away from their well-known theoretical
     values; the reference table below is measured directly against this
     module's own cumulant_features() applied to properly recovered symbols,
     for the same reason the original per-sample-convention note explained.

Because of (2), this function expects TWO inputs in the pipeline: the
continuous baseband IQ (for the envelope check) and the decimated symbol
stream (for cumulant classification) -- see classify_modulation()'s signature
and the orchestrator's Stage 7/8 ordering.
"""
import numpy as np

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
    """Valid ONLY on a continuous, pulse-shaped (not symbol-decimated) baseband
    signal -- see the module design note above for why unshaped signals break
    this check."""
    mag = np.abs(continuous_iq)
    return float(np.std(mag) / (np.mean(mag) + 1e-15)) < tolerance

# Reference |C40|,|C42| magnitudes, measured from this module's own
# cumulant_features() applied to properly RRC-shaped, matched-filtered, and
# symbol-rate-decimated signals (25dB SNR) -- NOT raw oversampled samples and
# NOT unshaped rectangular pulses. Recompute if the signal chain upstream of
# this stage changes.
_REFERENCE = {
    "bpsk":  {"c40": 1.49, "c42": 1.57},
    "qpsk":  {"c40": 0.76, "c42": 1.00},
    "16qam": {"c40": 0.61, "c42": 0.69},
}

def classify_modulation(continuous_iq, decimated_symbols=None):
    """continuous_iq: matched-filtered baseband, pre-timing-recovery (used for
    the FSK envelope check). decimated_symbols: post-Gardner symbol-rate
    samples (used for cumulant classification); if None, falls back to using
    continuous_iq for both (less accurate, but keeps the function usable
    standalone before timing recovery has run)."""
    if is_constant_envelope(continuous_iq):
        return {"modulation": "2fsk", "confidence": 0.85,
                "evidence": "near-constant envelope on the continuous waveform"}
    symbols_for_cumulants = decimated_symbols if decimated_symbols is not None else continuous_iq
    feats = cumulant_features(symbols_for_cumulants)
    best, best_dist = None, float("inf")
    for name, ref in _REFERENCE.items():
        dist = (feats["|c40|"] - ref["c40"]) ** 2 + (feats["|c42|"] - ref["c42"]) ** 2
        if dist < best_dist:
            best, best_dist = name, dist
    confidence = float(np.exp(-best_dist))
    return {"modulation": best, "confidence": confidence, "evidence": feats}


def classify_modulation_auto(continuous_iq, decimated_symbols=None, raw_iq=None):
    """Try CNN-based classification first (if trained model exists), then fall
    back to the cumulant-based classifier.

    Args:
        continuous_iq: matched-filtered baseband (for cumulant fallback)
        decimated_symbols: post-Gardner symbols (for cumulant fallback)
        raw_iq: original IQ samples for CNN input (1D complex64). If None,
                uses continuous_iq as the CNN input.

    Returns:
        dict with 'modulation', 'confidence', 'evidence'
    """
    cnn_input = raw_iq if raw_iq is not None else continuous_iq

    try:
        from rf_analyzer.amc.cnn_model import classify_iq_cnn
        cnn_result = classify_iq_cnn(cnn_input)
        if cnn_result["modulation"] is not None and cnn_result["confidence"] > 0.3:
            return cnn_result
    except Exception:
        pass  # CNN not available or failed -- fall back to cumulant

    return classify_modulation(continuous_iq, decimated_symbols)
