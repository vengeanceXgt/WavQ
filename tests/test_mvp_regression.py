import numpy as np
import os
import json
from rf_analyzer.orchestrator.pipeline import analyze_signal
from scratch.test_carrier_recovery_accuracy import generate_carrier_test_signal

def create_synthetic_file(filename, mod):
    iq, _, _, _ = generate_carrier_test_signal(mod, snr_db=15.0, cfo_hz=0.0)
    # Save as simple complex64 binary
    iq.astype(np.complex64).tofile(filename)
    return filename

def test_regression():
    print("--- MVP REGRESSION TEST ---")
    
    os.makedirs("data/test", exist_ok=True)
    f_bpsk = create_synthetic_file("data/test/bpsk.iq", "bpsk")
    f_qpsk = create_synthetic_file("data/test/qpsk.iq", "qpsk")
    f_16qam = create_synthetic_file("data/test/16qam.iq", "16qam")
    f_intelsat = r"D:\Aukaat\dataset_cache\datasets\intelsat37e\gr4-packet-modem-intelsat37e-test.sigmf-data"
    
    files = [
        (f_bpsk, "BPSK", {"sample_rate": 1.0, "sps": 4}),
        (f_qpsk, "QPSK", {"sample_rate": 1.0, "sps": 4}),
        (f_16qam, "16QAM", {"sample_rate": 1.0, "sps": 4}),
        (f_intelsat, "Intelsat", {"sample_rate": 2000000}) # Assuming 2Mhz just to parse
    ]
    
    for f, name, overrides in files:
        print(f"\nEvaluating: {name} -> {f}")
        res = analyze_signal(f, overrides)
        status = res.get("status")
        mod = res.get("modulation", {}).get("label")
        ambig = res.get("modulation", {}).get("ambiguous")
        print(f"Status: {status}")
        print(f"Detected Mod: {mod.upper() if mod else 'NONE'} (Ambiguous: {ambig})")
        print(f"Limitations: {res.get('limitations')}")
        assert status in ["complete", "partial"], "Pipeline completely failed!"

if __name__ == "__main__":
    test_regression()
