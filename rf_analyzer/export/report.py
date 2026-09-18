"""Export a full audit-trail report of a pipeline run."""
import json
import datetime
import numpy as np

def _json_safe(obj):
    """Recursively converts numpy types to plain Python types that json.dump()
    can handle. Python's json module natively understands its own str/int/
    float/bool/list/dict, but NOT numpy's own versions of those types (np.bool_,
    np.integer, np.floating) or complex numbers -- each raises a TypeError the
    first time a stage's data happens to contain one (e.g. a numpy bool from a
    comparison, or a complex cumulant value from the AMC stage's evidence)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (complex, np.complexfloating)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

def export_report(stages, source_file, out_path):
    report = {
        "source_file": str(source_file),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stages": [
            {"name": s.name, "ok": s.ok, "confidence": s.confidence,
             "data": _json_safe(s.data), "error": s.error}
            for s in stages
        ],
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path
