import numpy as np
from rf_analyzer.utils.synth import generate_test_signal
from rf_analyzer.ingest.sigmf_io import write_sigmf_meta
from rf_analyzer.orchestrator.pipeline import run_pipeline

def _write_test_file(tmp_path, **kwargs):
    iq, truth = generate_test_signal(**kwargs)
    path = tmp_path / "sig.iq"
    iq.tofile(path)
    write_sigmf_meta(path, sample_rate=truth["sample_rate"])
    return path, truth

def test_SYS01_end_to_end_qpsk_no_crash(tmp_path):
    path, truth = _write_test_file(tmp_path, scheme="qpsk", snr_db=20, cfo_hz=1000.0)
    stages = run_pipeline(path)
    by_name = {s.name: s for s in stages}
    assert all(s.ok for s in stages), [s.name for s in stages if not s.ok]
    assert by_name["amc"].data["modulation"] == "qpsk"

def test_SYS02_all_mvp_modulations_run_without_exception(tmp_path):
    for scheme in ["2fsk", "bpsk", "qpsk", "16qam"]:
        p = tmp_path / f"{scheme}"
        p.mkdir()
        cfo_hz = 0.0 if scheme == "2fsk" else 1000.0
        path, truth = _write_test_file(p, scheme=scheme, snr_db=20, cfo_hz=cfo_hz)
        stages = run_pipeline(path)
        by_name = {s.name: s for s in stages}
        assert by_name["ingest"].ok
        assert by_name["amc"].ok
        assert by_name["amc"].data["modulation"] == scheme, \
            f"{scheme} misclassified as {by_name['amc'].data['modulation']}"

def test_SYS03_graceful_degradation_on_corrupt_file(tmp_path):
    path = tmp_path / "bad.iq"
    path.write_bytes(b"\x00" * 13)
    stages = run_pipeline(path)
    assert stages[0].name == "ingest"
    assert stages[0].ok is False
    assert stages[0].error is not None

def test_SYS04_manual_override_is_respected(tmp_path):
    path, truth = _write_test_file(tmp_path, scheme="qpsk", snr_db=20)
    stages = run_pipeline(path, manual_overrides={
        "modulation": {"modulation": "16qam", "confidence": 1.0, "evidence": "manual"}
    })
    by_name = {s.name: s for s in stages}
    assert by_name["amc"].data["modulation"] == "16qam"

def test_SYS05_full_stage_list_present(tmp_path):
    """Confirms the previously-missing matched-filter stage is present and
    ordered correctly relative to sync and amc."""
    path, truth = _write_test_file(tmp_path, scheme="qpsk", snr_db=20)
    stages = run_pipeline(path)
    names = [s.name for s in stages]
    assert "matched_filter" in names
    assert names.index("matched_filter") < names.index("sync") < names.index("amc")

def test_SYS06_randomized_stress_test():
    """WHAT: the full pipeline across many randomized conditions (modulation,
    CFO, SNR, seed) rather than a handful of fixed cases.
    WHY: fixed test cases can accidentally avoid a bug that only shows up for
    certain parameter combinations. Broad randomized testing is a deliberate
    defense against that.
    EXPECTED: at least 27 of 30 random trials correctly classify modulation."""
    import tempfile, os
    rng = np.random.default_rng(0)
    hits = 0
    n_trials = 30
    with tempfile.TemporaryDirectory() as tmpdir:
        for trial in range(n_trials):
            scheme = str(rng.choice(["2fsk", "bpsk", "qpsk", "16qam"]))
            cfo = 0.0 if scheme == "2fsk" else float(rng.uniform(-4000, 4000))
            snr = float(rng.uniform(15, 25))
            seed = int(rng.integers(0, 100000))
            iq, truth = generate_test_signal(scheme=scheme, snr_db=snr, cfo_hz=cfo, seed=seed)
            path = os.path.join(tmpdir, f"stress_{trial}.iq")
            iq.tofile(path)
            write_sigmf_meta(path, sample_rate=truth["sample_rate"])
            stages = run_pipeline(path)
            by_name = {s.name: s for s in stages}
            if by_name["amc"].ok and by_name["amc"].data["modulation"] == scheme:
                hits += 1
    assert hits >= 27, f"only {hits}/{n_trials} randomized trials passed"
