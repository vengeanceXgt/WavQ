import numpy as np
import pytest
from rf_analyzer.utils.synth import generate_test_signal
from rf_analyzer.ingest.sigmf_io import write_sigmf_meta
from rf_analyzer.ingest.characterize import load_file, normalize

@pytest.fixture
def synthetic_iq_file(tmp_path):
    iq, truth = generate_test_signal(scheme="qpsk", sample_rate=1e6)
    path = tmp_path / "test.iq"
    iq.tofile(path)
    write_sigmf_meta(path, sample_rate=truth["sample_rate"])
    return path, truth

def test_ING01_load_sigmf_paired_iq(synthetic_iq_file):
    path, truth = synthetic_iq_file
    result = load_file(path)
    assert result.sample_rate == truth["sample_rate"]
    assert len(result.iq) > 0
    assert result.iq.dtype == np.complex64

def test_ING02_load_raw_iq_without_metadata(tmp_path):
    iq, _ = generate_test_signal(sample_rate=1e6)
    path = tmp_path / "no_meta.iq"
    iq.tofile(path)
    result = load_file(path)
    assert len(result.iq) > 0
    assert any("inferred" in n.lower() for n in result.notes)

def test_ING03_corrupted_file_size_raises_clear_error(tmp_path):
    path = tmp_path / "corrupt.iq"
    path.write_bytes(b"\x00" * 13)
    with pytest.raises(ValueError, match="not a multiple"):
        load_file(path)

def test_ING04_normalize_removes_dc_and_unit_power():
    iq, _ = generate_test_signal()
    iq_biased = iq + (5 + 2j)
    normed, clipped = normalize(iq_biased)
    assert abs(np.mean(normed)) < 0.05
    assert abs(np.mean(np.abs(normed) ** 2) - 1.0) < 0.05
    assert clipped is False  # normalize() now returns a plain python bool
