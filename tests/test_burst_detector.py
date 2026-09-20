import numpy as np
from rf_analyzer.spectral.burst_detector import detect_bursts

def _generate_noise(length, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(scale=1.0, size=length) + 1j * rng.normal(scale=1.0, size=length)).astype(np.complex64)

def _add_burst(iq, start, end, amplitude=10.0):
    iq[start:end] += amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi, size=(end-start)))
    return iq

def test_burst_pure_noise():
    iq = _generate_noise(10000)
    bursts = detect_bursts(iq, sample_rate=1e6)
    assert len(bursts) == 0

def test_burst_single_known():
    iq = _generate_noise(10000)
    iq = _add_burst(iq, 3000, 7000)
    bursts = detect_bursts(iq, sample_rate=1e6)
    assert len(bursts) == 1
    b = bursts[0]
    # Check boundaries with some tolerance due to smoothing
    assert abs(b.start_sample - 3000) <= 20
    assert abs(b.end_sample - 7000) <= 20

def test_burst_separated():
    iq = _generate_noise(20000)
    iq = _add_burst(iq, 2000, 6000)
    iq = _add_burst(iq, 12000, 16000)
    bursts = detect_bursts(iq, sample_rate=1e6)
    assert len(bursts) == 2
    assert abs(bursts[0].start_sample - 2000) <= 20
    assert abs(bursts[1].start_sample - 12000) <= 20

def test_burst_gap_merging():
    iq = _generate_noise(20000)
    iq = _add_burst(iq, 2000, 6000)
    # Gap of 500 samples (0.5 ms at 1e6)
    iq = _add_burst(iq, 6500, 10000)
    
    # min_gap_s = 0.001 (1 ms = 1000 samples)
    bursts = detect_bursts(iq, sample_rate=1e6, min_gap_s=0.001)
    assert len(bursts) == 1
    assert abs(bursts[0].start_sample - 2000) <= 20
    assert abs(bursts[0].end_sample - 10000) <= 20

def test_burst_fading_hysteresis():
    iq = _generate_noise(10000)
    # create a fading envelope
    t = np.linspace(0, np.pi, 4000)
    envelope = 8.0 * np.sin(t) + 3.0 * np.sin(5*t) # Adds variation
    burst_iq = envelope * np.exp(1j * np.random.uniform(0, 2*np.pi, size=4000))
    iq[3000:7000] += burst_iq
    
    bursts = detect_bursts(iq, sample_rate=1e6, threshold_on_db=10.0, threshold_off_db=3.0)
    assert len(bursts) == 1
    # Check that fading didn't break it into pieces
    assert bursts[0].start_sample > 2900
    assert bursts[0].end_sample < 7100

def test_burst_very_short():
    iq = _generate_noise(10000)
    # Add a 50 sample burst (0.05 ms)
    iq = _add_burst(iq, 5000, 5050, amplitude=20.0)
    
    # min_duration_s = 0.001 (1 ms)
    bursts = detect_bursts(iq, sample_rate=1e6, min_duration_s=0.001)
    assert len(bursts) == 0 # Should be filtered out

def test_burst_continuous():
    iq = _generate_noise(10000)
    # Burst covers 90% of the file
    iq = _add_burst(iq, 500, 9500)
    bursts = detect_bursts(iq, sample_rate=1e6)
    assert len(bursts) == 1
    assert bursts[0].start_sample <= 510
    assert bursts[0].end_sample >= 9490
