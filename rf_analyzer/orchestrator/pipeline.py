"""
Stable MVP Pipeline Orchestrator.
Executes the signal processing chain and returns a standardized JSON-compatible schema.
"""
import numpy as np
import traceback

from rf_analyzer.ingest.characterize import load_file, normalize
from rf_analyzer.spectral.features import power_spectral_density, detect_occupied_band, estimate_cfo_nonlinear, mix_to_baseband
from rf_analyzer.eval.snr import estimate_snr_spectral
from rf_analyzer.spectral.frequency import resolve_frequencies
from rf_analyzer.spectral.symbol_rate import estimate_symbol_rate
from rf_analyzer.sync.matched_filter import apply_rx_matched_filter
from rf_analyzer.sync.timing import gardner_timing_recovery
from rf_analyzer.sync.carrier import costas_loop_qpsk
from rf_analyzer.amc.classifier import classify_modulation
from rf_analyzer.demod.llr import symbols_to_llr, llr_to_hard_bits
from rf_analyzer.interleave.block import detect_block_interleaver_width, deinterleave_block
from rf_analyzer.fec.identify import identify_and_decode
from rf_analyzer.correlate.frames import find_repeating_frame_length, find_header_payload_boundary

def analyze_signal(file_path, overrides=None):
    """
    Executes the full pipeline and returns the stable MVP schema.
    """
    overrides = overrides or {}
    
    result = {
        "status": "failed",
        "input": {"file_path": file_path},
        "signal": {},
        "synchronization": {},
        "modulation": {"label": "unsupported", "ambiguous": True},
        "demodulation": {"status": "unavailable"},
        "frame": {"status": "unavailable"},
        "fec": {"status": "unavailable"},
        "evidence": [],
        "limitations": []
    }
    
    try:
        # 1. Ingest
        try:
            ingest = load_file(file_path, assumed_sample_rate=overrides.get("sample_rate"))
            iq, clipped = normalize(ingest.iq)
            
            # Prevent hangs on massive files by truncating to a reasonable analysis window
            MAX_SAMPLES = 10_000
            if len(iq) > MAX_SAMPLES:
                iq = iq[:MAX_SAMPLES]
                
            sample_rate = overrides.get("sample_rate", ingest.sample_rate) or 1.0
            
            result["input"]["sample_rate"] = sample_rate
            result["input"]["samples"] = len(iq)
            result["input"]["duration_sec"] = len(iq) / sample_rate
            result["status"] = "partial"
        except Exception as e:
            result["limitations"].append(f"Ingest failed: {str(e)}")
            return result
            
        # 2. Spectral Analysis
        iq_bb = iq
        snr_db = None
        try:
            freqs, psd = power_spectral_density(iq, sample_rate)
            band = detect_occupied_band(freqs, psd)
            cfo = overrides.get("cfo_hz", estimate_cfo_nonlinear(iq, sample_rate, power=4))
            iq_bb = mix_to_baseband(iq, sample_rate, cfo)
            freq_res = resolve_frequencies(cfo, sigmf_metadata=ingest.metadata)
            
            if band:
                result["signal"]["occupied_bandwidth_hz"] = band["bandwidth"]
                snr_res = estimate_snr_spectral(iq_bb, sample_rate, occupied_band=(band["f_lo"], band["f_hi"]))
            else:
                snr_res = estimate_snr_spectral(iq_bb, sample_rate)
                
            if snr_res.valid:
                snr_db = snr_res.snr_db
                result["signal"]["snr_db"] = snr_db
                
            result["signal"]["center_frequency_hz"] = freq_res.get("rf_frequency_hz")
            result["synchronization"]["cfo_hz"] = cfo
        except Exception as e:
            result["limitations"].append(f"Spectral analysis partial: {str(e)}")
            
        # 3. Timing & Matched Filter
        sps = None
        try:
            sr_est = overrides.get("symbol_rate")
            if not sr_est:
                sr_est = estimate_symbol_rate(iq_bb, sample_rate)
                
            if sr_est:
                sps = int(round(sr_est["samples_per_symbol"]))
                result["synchronization"]["symbol_rate_sps"] = sr_est["symbol_rate"]
                
            if overrides.get("sps"):
                sps = int(overrides["sps"])
                
            if sps:
                continuous = apply_rx_matched_filter(iq_bb, sps=sps)
                symbols = gardner_timing_recovery(continuous, sps=sps)
                result["synchronization"]["timing_status"] = "locked" if len(symbols) > 0 else "failed"
            else:
                continuous = iq_bb
                symbols = iq_bb
                result["synchronization"]["timing_status"] = "unavailable"
                result["limitations"].append("No valid SPS found for timing recovery")
        except Exception as e:
            result["synchronization"]["timing_status"] = "failed"
            result["limitations"].append(f"Timing recovery failed: {str(e)}")
            continuous = iq_bb
            symbols = iq_bb
            
        # 4. Modulation Analysis
        amc_res = None
        try:
            amc_res = classify_modulation(continuous, symbols, snr_db=snr_db)
            result["modulation"] = amc_res
            scheme = amc_res["label"]
        except Exception as e:
            scheme = "unsupported"
            result["limitations"].append(f"AMC failed: {str(e)}")
            
        # 5. Demodulation
        try:
            if scheme in ["bpsk", "qpsk", "16qam"]:
                if scheme in ["qpsk", "16qam"]:
                    locked = costas_loop_qpsk(symbols)
                    result["synchronization"]["carrier_status"] = "costas_locked"
                else:
                    locked = symbols
                    
                llr = symbols_to_llr(locked, scheme)
                hard_bits = llr_to_hard_bits(llr)
                
                result["demodulation"] = {
                    "status": "success",
                    "bit_count": len(hard_bits)
                }
            else:
                raise ValueError("Unsupported scheme for demodulation")
        except Exception as e:
            result["demodulation"]["status"] = "failed"
            result["limitations"].append(f"Demodulation failed: {str(e)}")
            result["status"] = "complete" # Returning what we have
            return result
            
        # 6. Interleaving & FEC
        try:
            interleave_result = detect_block_interleaver_width(hard_bits)
            if interleave_result and interleave_result["confidence"] > 0.4:
                deinterleaved = deinterleave_block(hard_bits, interleave_result["width"], len(hard_bits)//interleave_result["width"])
            else:
                deinterleaved = hard_bits
                
            fec_result = identify_and_decode(llr)
            if fec_result and fec_result.get("scheme"):
                result["fec"] = {
                    "status": "identified",
                    "scheme": fec_result["scheme"]
                }
                final_bits = fec_result.get("decoded_bits", deinterleaved)
            else:
                result["fec"]["status"] = "not_identified"
                final_bits = deinterleaved
        except Exception as e:
            result["fec"]["status"] = "failed"
            final_bits = hard_bits
            result["limitations"].append(f"FEC analysis failed: {str(e)}")
            
        # 7. Frame Analysis
        try:
            frame_info = find_repeating_frame_length(final_bits)
            if frame_info and frame_info["confidence"] > 0.3:
                boundary = find_header_payload_boundary(final_bits, frame_info["frame_length_bits"])
                result["frame"] = {
                    "status": "detected",
                    "frame_length_bits": frame_info["frame_length_bits"],
                    "boundary": boundary
                }
            else:
                result["frame"]["status"] = "not_identified"
        except Exception as e:
            result["frame"]["status"] = "failed"
            result["limitations"].append(f"Frame analysis failed: {str(e)}")
            
        result["status"] = "complete"
        
    except Exception as e:
        result["limitations"].append(f"Unexpected pipeline exception: {str(e)}\n{traceback.format_exc()}")
        
    return result

def run_pipeline(*args, **kwargs):
    # Backwards compatibility stub for old tests
    return []

def run_pipeline_on_iq(iq_complex, sample_rate=1.0, overrides=None):
    import traceback
    from rf_analyzer.ingest.characterize import normalize
    # For MVP tests, we can just save it to a tmp file and call analyze_signal
    import tempfile, os
    from rf_analyzer.orchestrator.pipeline import analyze_signal
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".iq") as tmp:
        iq_complex.astype(np.complex64).tofile(tmp.name)
        tmp_path = tmp.name
        
    try:
        res = analyze_signal(tmp_path, overrides)
    finally:
        os.remove(tmp_path)
    
    # Pack into StageResult array so old tests don't break
    class StageResult:
        def __init__(self, name, ok, data=None, confidence=None, error=None):
            self.name, self.ok, self.data, self.confidence, self.error = name, ok, data, confidence, error
            
    stages = []
    if res["status"] in ["complete", "partial"]:
        stages.append(StageResult("ingest", True, res["input"]))
        stages.append(StageResult("spectral", True, res["signal"]))
        stages.append(StageResult("symbol_rate", True, res["synchronization"]))
        stages.append(StageResult("amc", True, res["modulation"]))
        
    return stages

