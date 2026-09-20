import numpy as np
from rf_analyzer.utils.synth import generate_test_signal
from rf_analyzer.orchestrator.pipeline import run_pipeline_on_iq
from rf_analyzer.eval.snr import estimate_snr_m2m4
from rf_analyzer.spectral.features import estimate_cfo_nonlinear

def run_impaired_test(scheme="qpsk", snr_db=15.0, cfo_hz=0.0, timing_offset=0.0, phase_offset=0.0, fading=False, seed=42):
    iq, truth = generate_test_signal(scheme=scheme, sps=8, snr_db=snr_db, cfo_hz=cfo_hz, sample_rate=1e6, seed=seed)
    
    rng = np.random.default_rng(seed)
    
    # 1. Apply Phase Offset
    iq = iq * np.exp(1j * phase_offset)
    
    # 2. Apply Fading (simple sinusoidal envelope for bursty fading)
    if fading:
        t = np.arange(len(iq))
        envelope = 0.5 * (1 + np.sin(2 * np.pi * t / (len(iq) / 4)))  # 4 slow fades
        iq = iq * envelope
        
    # 3. Apply timing offset (fractional delay via phase shift in freq domain)
    if timing_offset != 0.0:
        N = len(iq)
        freqs = np.fft.fftfreq(N)
        iq_f = np.fft.fft(iq)
        iq_f = iq_f * np.exp(-1j * 2 * np.pi * freqs * timing_offset)
        iq = np.fft.ifft(iq_f)
        
    # Run the estimators
    est_cfo = estimate_cfo_nonlinear(iq, sample_rate=1e6, power=4)
    
    # Mix to baseband for sync
    t = np.arange(len(iq)) / 1e6
    iq_bb = iq * np.exp(-1j * 2 * np.pi * est_cfo * t)
    
    # SNR
    snr_res = estimate_snr_m2m4(iq_bb)
    
    return {
        "true_cfo": cfo_hz,
        "est_cfo": float(est_cfo),
        "true_snr": float(snr_db),
        "est_snr": float(snr_res.snr_db) if snr_res.valid else None,
        "cfo_error": float(abs(cfo_hz - est_cfo))
    }

def test_impaired_cfo():
    res = run_impaired_test(cfo_hz=5000.0)
    print("CFO Test:", res)
    assert res["cfo_error"] < 100.0

def test_impaired_fading():
    res = run_impaired_test(fading=True)
    print("Fading Test:", res)

def test_impaired_timing_offset():
    res = run_impaired_test(timing_offset=3.5) # 3.5 samples delay
    print("Timing Offset Test:", res)

if __name__ == "__main__":
    test_impaired_cfo()
    test_impaired_fading()
    test_impaired_timing_offset()
