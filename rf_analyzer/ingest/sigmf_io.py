"""Read/write helpers around the SigMF standard (sample rate + datatype metadata)."""
import json
import numpy as np
from pathlib import Path

DTYPE_MAP = {
    "cf32_le": np.complex64,
    "cf64_le": np.complex128,
    "ci16_le": np.int16,   # interleaved I/Q int16, needs de-interleave + scale
}

def load_sigmf(data_path):
    """Loads a .sigmf-data file using its sibling .sigmf-meta if present."""
    data_path = Path(data_path)
    meta_path = data_path.with_suffix(".sigmf-meta")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        dtype_str = meta["global"]["core:datatype"]
        sample_rate = meta["global"].get("core:sample_rate", None)
    else:
        meta, dtype_str, sample_rate = None, None, None
    raw = np.fromfile(data_path, dtype=np.uint8)
    return raw, dtype_str, sample_rate, meta

def write_sigmf_meta(data_path, sample_rate, datatype="cf32_le", extra=None):
    """Writes a .sigmf-meta JSON sidecar next to data_path."""
    data_path = Path(data_path)
    meta = {
        "global": {
            "core:datatype": datatype,
            "core:sample_rate": sample_rate,
            "core:version": "1.0.0",
        },
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }
    if extra:
        meta["global"].update(extra)
    meta_path = data_path.with_suffix(".sigmf-meta")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path
