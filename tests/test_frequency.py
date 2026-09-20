from rf_analyzer.spectral.frequency import resolve_frequencies

def test_freq_manual_override():
    res = resolve_frequencies(100.0, manual_rf_hz=1e9)
    assert res.absolute_rf_hz == 1e9 + 100.0
    assert res.source == "user"

def test_freq_sigmf_metadata():
    meta = {"captures": [{"core:frequency": 2.4e9}]}
    res = resolve_frequencies(-500.0, sigmf_metadata=meta)
    assert res.absolute_rf_hz == 2.4e9 - 500.0
    assert res.source == "metadata"

def test_freq_unavailable():
    res = resolve_frequencies(250.0)
    assert res.absolute_rf_hz is None
    assert res.source == "unavailable"
    assert "unavailable" in res.reason
