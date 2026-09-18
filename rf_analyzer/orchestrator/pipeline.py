"""Runs the full pipeline end-to-end, tracking confidence at every stage.

STAGE ORDER FIX: an earlier version ran AMC directly after carrier mixing, on
raw oversampled IQ, before timing recovery. That was found to be wrong once
tested against realistically pulse-shaped signals -- classification needs
either the continuous matched-filtered waveform (for the FSK envelope check)
or properly decimated symbol-rate samples (for cumulant classification), not
raw oversampled IQ. The corrected order is: ingest -> spectral (CFO/band) ->
RX matched filter -> timing recovery (Gardner) -> AMC (on the matched-filtered
continuous signal AND the recovered symbols) -> carrier tracking (Costas) ->
demod -> interleave -> FEC -> correlate. Timing recovery does not need to know
the modulation scheme, so it can safely run before AMC.
"""
import numpy as np
from rf_analyzer.ingest.characterize import load_file, normalize
from rf_analyzer.spectral.features import (power_spectral_density, detect_occupied_band,
                                            estimate_cfo_nonlinear, mix_to_baseband)
from rf_analyzer.spectral.symbol_rate import estimate_symbol_rate
from rf_analyzer.sync.matched_filter import apply_rx_matched_filter
from rf_analyzer.sync.timing import gardner_timing_recovery
from rf_analyzer.sync.carrier import costas_loop_qpsk
from rf_analyzer.amc.classifier import classify_modulation, classify_modulation_auto
from rf_analyzer.demod.llr import symbols_to_llr, llr_to_hard_bits
from rf_analyzer.interleave.block import detect_block_interleaver_width, deinterleave_block
from rf_analyzer.fec.identify import identify_and_decode
from rf_analyzer.correlate.frames import find_repeating_frame_length, find_header_payload_boundary

class StageResult:
    def __init__(self, name, ok, data=None, confidence=None, error=None):
        self.name, self.ok, self.data = name, ok, data
        self.confidence, self.error = confidence, error

def run_pipeline(file_path, assumed_sample_rate=None, manual_overrides=None, use_cnn_amc=False):
    manual_overrides = manual_overrides or {}
    stages = []

    def record(name, ok, data=None, confidence=None, error=None):
        stages.append(StageResult(name, ok, data, confidence, error))

    try:
        ingest = load_file(file_path, assumed_sample_rate=assumed_sample_rate)
        iq, clipped = normalize(ingest.iq)
        record("ingest", True, {"sample_rate": ingest.sample_rate,
                                 "notes": ingest.notes, "clipped": clipped})
    except Exception as e:
        record("ingest", False, error=str(e))
        return stages

    sample_rate = manual_overrides.get("sample_rate", ingest.sample_rate) or 1.0

    try:
        freqs, psd = power_spectral_density(iq, sample_rate)
        band = detect_occupied_band(freqs, psd)
        cfo = manual_overrides.get("cfo_hz", estimate_cfo_nonlinear(iq, sample_rate, power=4))
        iq_bb = mix_to_baseband(iq, sample_rate, cfo)
        record("spectral", band is not None, {"band": band, "cfo_hz": cfo},
               confidence=0.7 if band else 0.0)
    except Exception as e:
        record("spectral", False, error=str(e))
        return stages

    try:
        sr_est = manual_overrides.get("symbol_rate", estimate_symbol_rate(iq_bb, sample_rate))
        record("symbol_rate", sr_est is not None, sr_est,
               confidence=0.6 if sr_est else 0.0)
        sps = int(round(sr_est["samples_per_symbol"])) if sr_est else 8
    except Exception as e:
        record("symbol_rate", False, error=str(e))
        sps = 8

    try:
        continuous = apply_rx_matched_filter(iq_bb, sps=sps)
        record("matched_filter", True, {"n_samples": len(continuous)}, confidence=None)
    except Exception as e:
        record("matched_filter", False, error=str(e))
        return stages

    try:
        symbols = gardner_timing_recovery(continuous, sps=sps)
        record("sync", len(symbols) > 0, {"n_symbols": len(symbols)},
               confidence=0.6 if len(symbols) > 0 else 0.0)
    except Exception as e:
        record("sync", False, error=str(e))
        return stages

    try:
        if manual_overrides.get("modulation"):
            amc = manual_overrides["modulation"]
        elif use_cnn_amc:
            amc = classify_modulation_auto(continuous, symbols, raw_iq=iq)
        else:
            amc = classify_modulation(continuous, symbols)
        record("amc", True, amc, confidence=amc.get("confidence"))
        scheme = amc["modulation"] if isinstance(amc, dict) else amc
    except Exception as e:
        record("amc", False, error=str(e))
        return stages

    try:
        locked = costas_loop_qpsk(symbols) if scheme in ("qpsk", "16qam") else symbols
    except Exception as e:
        record("carrier_track", False, error=str(e))
        locked = symbols

    try:
        llr = symbols_to_llr(locked, scheme)
        record("demod", True, {"n_llr": len(llr)}, confidence=0.6)
    except Exception as e:
        record("demod", False, error=str(e))
        return stages

    try:
        hard_bits = llr_to_hard_bits(llr)
        interleave_result = detect_block_interleaver_width(hard_bits)
        record("interleave_detect", interleave_result is not None, interleave_result,
               confidence=interleave_result["confidence"] if interleave_result else 0.0)
        if interleave_result and interleave_result["confidence"] > 0.4:
            deinterleaved_bits = deinterleave_block(
                hard_bits, interleave_result["width"],
                len(hard_bits) // interleave_result["width"])
        else:
            deinterleaved_bits = hard_bits
        llr_for_fec = llr
    except Exception as e:
        record("interleave_detect", False, error=str(e))
        deinterleaved_bits, llr_for_fec = hard_bits, llr

    try:
        fec_result = identify_and_decode(llr_for_fec)
        record("fec", True, fec_result, confidence=fec_result.get("confidence"))
        final_bits = fec_result.get("decoded_bits")
        if final_bits is None:
            final_bits = deinterleaved_bits
    except Exception as e:
        record("fec", False, error=str(e))
        final_bits = deinterleaved_bits

    try:
        frame_info = find_repeating_frame_length(final_bits)
        boundary_info = None
        if frame_info and frame_info["confidence"] > 0.3:
            boundary_info = find_header_payload_boundary(final_bits, frame_info["frame_length_bits"])
        record("correlate", frame_info is not None, {"frame": frame_info, "boundary": boundary_info},
               confidence=frame_info["confidence"] if frame_info else 0.0)
    except Exception as e:
        record("correlate", False, error=str(e))

    return stages


def run_pipeline_on_iq(iq_complex, sample_rate=1.0, manual_overrides=None, use_cnn_amc=False):
    """Run the pipeline directly on an IQ array (no file I/O).

    This is useful for evaluating against datasets like RadioML 2018 where
    the data is already loaded as complex numpy arrays.

    Args:
        iq_complex: 1D complex64 numpy array of IQ samples
        sample_rate: sample rate in Hz (default 1.0 for normalized frequency)
        manual_overrides: dict of manual stage overrides
        use_cnn_amc: if True, try CNN-based AMC first

    Returns:
        list of StageResult objects
    """
    manual_overrides = manual_overrides or {}
    stages = []

    def record(name, ok, data=None, confidence=None, error=None):
        stages.append(StageResult(name, ok, data, confidence, error))

    # Ingest: normalize directly from array
    try:
        from rf_analyzer.ingest.characterize import normalize
        iq, clipped = normalize(iq_complex.astype(np.complex64))
        record("ingest", True, {"sample_rate": sample_rate,
                                 "notes": ["Direct IQ array input"], "clipped": clipped})
    except Exception as e:
        record("ingest", False, error=str(e))
        return stages

    sample_rate = manual_overrides.get("sample_rate", sample_rate) or 1.0

    try:
        freqs, psd = power_spectral_density(iq, sample_rate)
        band = detect_occupied_band(freqs, psd)
        cfo = manual_overrides.get("cfo_hz", estimate_cfo_nonlinear(iq, sample_rate, power=4))
        iq_bb = mix_to_baseband(iq, sample_rate, cfo)
        record("spectral", band is not None, {"band": band, "cfo_hz": cfo},
               confidence=0.7 if band else 0.0)
    except Exception as e:
        record("spectral", False, error=str(e))
        return stages

    try:
        sr_est = manual_overrides.get("symbol_rate", estimate_symbol_rate(iq_bb, sample_rate))
        record("symbol_rate", sr_est is not None, sr_est,
               confidence=0.6 if sr_est else 0.0)
        sps = int(round(sr_est["samples_per_symbol"])) if sr_est else 8
    except Exception as e:
        record("symbol_rate", False, error=str(e))
        sps = 8

    try:
        continuous = apply_rx_matched_filter(iq_bb, sps=sps)
        record("matched_filter", True, {"n_samples": len(continuous)}, confidence=None)
    except Exception as e:
        record("matched_filter", False, error=str(e))
        return stages

    try:
        symbols = gardner_timing_recovery(continuous, sps=sps)
        record("sync", len(symbols) > 0, {"n_symbols": len(symbols)},
               confidence=0.6 if len(symbols) > 0 else 0.0)
    except Exception as e:
        record("sync", False, error=str(e))
        return stages

    try:
        if manual_overrides.get("modulation"):
            amc = manual_overrides["modulation"]
        elif use_cnn_amc:
            amc = classify_modulation_auto(continuous, symbols, raw_iq=iq)
        else:
            amc = classify_modulation(continuous, symbols)
        record("amc", True, amc, confidence=amc.get("confidence"))
        scheme = amc["modulation"] if isinstance(amc, dict) else amc
    except Exception as e:
        record("amc", False, error=str(e))
        return stages

    try:
        locked = costas_loop_qpsk(symbols) if scheme in ("qpsk", "16qam") else symbols
    except Exception as e:
        record("carrier_track", False, error=str(e))
        locked = symbols

    try:
        llr = symbols_to_llr(locked, scheme)
        record("demod", True, {"n_llr": len(llr)}, confidence=0.6)
    except Exception as e:
        record("demod", False, error=str(e))
        return stages

    try:
        hard_bits = llr_to_hard_bits(llr)
        interleave_result = detect_block_interleaver_width(hard_bits)
        record("interleave_detect", interleave_result is not None, interleave_result,
               confidence=interleave_result["confidence"] if interleave_result else 0.0)
        if interleave_result and interleave_result["confidence"] > 0.4:
            deinterleaved_bits = deinterleave_block(
                hard_bits, interleave_result["width"],
                len(hard_bits) // interleave_result["width"])
        else:
            deinterleaved_bits = hard_bits
        llr_for_fec = llr
    except Exception as e:
        record("interleave_detect", False, error=str(e))
        deinterleaved_bits, llr_for_fec = hard_bits, llr

    try:
        fec_result = identify_and_decode(llr_for_fec)
        record("fec", True, fec_result, confidence=fec_result.get("confidence"))
        final_bits = fec_result.get("decoded_bits")
        if final_bits is None:
            final_bits = deinterleaved_bits
    except Exception as e:
        record("fec", False, error=str(e))
        final_bits = deinterleaved_bits

    try:
        frame_info = find_repeating_frame_length(final_bits)
        boundary_info = None
        if frame_info and frame_info["confidence"] > 0.3:
            boundary_info = find_header_payload_boundary(final_bits, frame_info["frame_length_bits"])
        record("correlate", frame_info is not None, {"frame": frame_info, "boundary": boundary_info},
               confidence=frame_info["confidence"] if frame_info else 0.0)
    except Exception as e:
        record("correlate", False, error=str(e))

    return stages
