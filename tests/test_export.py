import json
from rf_analyzer.utils.synth import generate_test_signal
from rf_analyzer.ingest.sigmf_io import write_sigmf_meta
from rf_analyzer.orchestrator.pipeline import run_pipeline
from rf_analyzer.export.report import export_report

def test_EXPORT01_report_roundtrips_all_stages(tmp_path):
    """WHAT: exporting a real pipeline run's results to JSON.
    WHY: numpy types (bool_, complex, various int/float widths) are NOT
    natively JSON-serializable and this has been a real, repeated bug source
    in this project -- this test must run against REAL pipeline output, not
    a hand-built dict, or it won't catch the gap (see Part 2's Export/Report
    troubleshooting section).
    EXPECTED: valid JSON file with every stage name from the pipeline run
    present in the exported report."""
    iq, truth = generate_test_signal(scheme="qpsk", snr_db=20, cfo_hz=1000.0)
    src = tmp_path / "src.iq"
    iq.tofile(src)
    write_sigmf_meta(src, sample_rate=truth["sample_rate"])
    stages = run_pipeline(src)
    out_path = tmp_path / "report.json"
    export_report(stages, src, out_path)

    with open(out_path) as f:
        report = json.load(f)  # raises if not valid JSON

    exported_names = {s["name"] for s in report["stages"]}
    real_names = {s.name for s in stages}
    assert exported_names == real_names
    assert report["source_file"] == str(src)

def test_EXPORT02_all_four_schemes_export_cleanly(tmp_path):
    """Runs export against every MVP modulation, since different stages'
    'evidence' data (e.g. AMC's complex cumulant values) only appear for
    certain code paths."""
    for scheme in ["2fsk", "bpsk", "qpsk", "16qam"]:
        p = tmp_path / scheme
        p.mkdir()
        cfo_hz = 0.0 if scheme == "2fsk" else 1000.0
        iq, truth = generate_test_signal(scheme=scheme, snr_db=20, cfo_hz=cfo_hz)
        src = p / "src.iq"
        iq.tofile(src)
        write_sigmf_meta(src, sample_rate=truth["sample_rate"])
        stages = run_pipeline(src)
        out_path = p / "report.json"
        export_report(stages, src, out_path)
        with open(out_path) as f:
            json.load(f)  # raises if not valid JSON
