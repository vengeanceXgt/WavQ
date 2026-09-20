"""Stage 1: File Characterization. Sniff format, coerce to normalized complex64 IQ."""
import numpy as np
import wave
from pathlib import Path
from scipy.signal import hilbert
from .sigmf_io import load_sigmf, DTYPE_MAP

class IngestResult:
    def __init__(self, iq, sample_rate, source_format, notes, metadata=None):
        self.iq = iq                    # complex64 numpy array
        self.sample_rate = sample_rate  # float Hz, or None if unknown
        self.source_format = source_format
        self.notes = notes              # list[str] of warnings/decisions made
        self.metadata = metadata        # dict of sigmf metadata if available

def _guess_raw_iq_dtype(raw_bytes):
    """Heuristic: try cf32 first (float exponent histogram is distinctive),
    fall back to 16-bit interleaved ints if the byte count doesn't divide evenly
    or if float interpretation produces infs/nans/huge values."""
    n = len(raw_bytes)
    if n % 8 == 0:
        # Check if it looks like float32
        floats = np.frombuffer(raw_bytes[:min(8000, n)], dtype=np.float32)
        if not np.all(np.isfinite(floats)):
            return "ci16_le"

        abs_val = np.abs(floats[floats != 0])
        if len(abs_val) > 0 and (np.max(abs_val) > 1e10 or np.min(abs_val) < 1e-30):
            return "ci16_le"

        return "cf32_le"
    if n % 4 == 0:
        return "ci16_le"
    return None  # unrecognized / truncated file

def load_file(path, assumed_sample_rate=None):
    path = Path(path)
    notes = []
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            sampwidth = wf.getsampwidth()
            n_channels = wf.getnchannels()
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sampwidth]
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if n_channels == 2:
            audio = audio.reshape(-1, 2)
            iq = (audio[:, 0] + 1j * audio[:, 1]).astype(np.complex64)
            notes.append("Stereo WAV interpreted directly as I/Q channels.")
        else:
            iq = hilbert(audio).astype(np.complex64)
            notes.append("Mono WAV: applied Hilbert transform to form analytic (I/Q) signal.")
        return IngestResult(iq, float(sample_rate), "wav", notes, metadata=None)

    # Not a .wav: try SigMF-paired or raw .iq
    raw, dtype_str, sample_rate, meta = load_sigmf(path)
    if dtype_str is None:
        dtype_str = _guess_raw_iq_dtype(raw)
        notes.append(f"No .sigmf-meta found; inferred datatype as {dtype_str} from file size.")
        if dtype_str is None:
            raise ValueError(
                f"File size {len(raw)} bytes is not a multiple of 4 or 8; "
                "cannot determine sample datatype. File may be corrupted or truncated."
            )
    if sample_rate is None:
        sample_rate = assumed_sample_rate
        if sample_rate is None:
            notes.append("Sample rate unknown: no SigMF metadata and none supplied. "
                          "Downstream stages will report normalized (0-0.5) digital "
                          "frequencies instead of absolute Hz until a rate is provided.")

    if dtype_str == "cf32_le":
        iq = raw.view(np.complex64)
    elif dtype_str == "ci16_le":
        ints = raw.view(np.int16).astype(np.float32) / 32768.0
        iq = (ints[0::2] + 1j * ints[1::2]).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported datatype: {dtype_str}")

    return IngestResult(iq, sample_rate, "iq", notes, metadata=meta)

def normalize(iq):
    """DC removal + unity variance normalization (Stage 1/2 signal conditioning)."""
    iq = iq - np.mean(iq)
    power = np.mean(np.abs(iq) ** 2)
    if power > 0:
        iq = iq / np.sqrt(power)
    clip_fraction = np.mean(np.abs(iq) > 0.98 * np.max(np.abs(iq))) if len(iq) else 0
    clipped = bool(clip_fraction > 0.001)
    return iq.astype(np.complex64), clipped
