import numpy as np
from rf_analyzer.correlate.frames import (find_repeating_frame_length, find_header_payload_boundary,
                                           crc16_ccitt, verify_crc16, bits_to_bytes)

def _build_framed_stream(n_frames=40, frame_len=128, sync_word_bits=16, seed=0):
    rng = np.random.default_rng(seed)
    sync = rng.integers(0, 2, size=sync_word_bits, dtype=np.uint8)
    frames = []
    for _ in range(n_frames):
        payload_len = frame_len - sync_word_bits - 16
        payload = rng.integers(0, 2, size=payload_len, dtype=np.uint8)
        payload_bytes = bits_to_bytes(payload)
        crc = crc16_ccitt(payload_bytes)
        crc_bits = np.array([int(b) for b in f"{crc:016b}"], dtype=np.uint8)
        frame = np.concatenate([sync, payload, crc_bits])
        frames.append(frame)
    return np.concatenate(frames), frame_len, sync_word_bits

def test_CORR01_detects_known_frame_length():
    bits, true_len, sync_len = _build_framed_stream()
    result = find_repeating_frame_length(bits, min_len=32, max_len=256)
    assert result is not None
    assert result["frame_length_bits"] == true_len

def test_CORR02_negative_case_unstructured_bits_low_confidence():
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, size=5000, dtype=np.uint8)
    result = find_repeating_frame_length(bits, min_len=32, max_len=256)
    assert result is None or result["confidence"] < 0.3

def test_CORR03_header_boundary_matches_sync_word_length():
    bits, frame_len, sync_len = _build_framed_stream()
    frame_info = find_repeating_frame_length(bits, min_len=32, max_len=256)
    boundary = find_header_payload_boundary(bits, frame_info["frame_length_bits"])
    assert boundary is not None
    assert boundary["payload_offset_bits"] == sync_len

def test_CORR04_crc_validates_correct_payload():
    bits, frame_len, sync_len = _build_framed_stream(n_frames=1)
    one_frame = bits[:frame_len]
    payload_plus_crc = one_frame[sync_len:]
    assert verify_crc16(payload_plus_crc) is True

def test_CORR05_crc_rejects_corrupted_payload():
    bits, frame_len, sync_len = _build_framed_stream(n_frames=1)
    one_frame = bits[:frame_len].copy()
    one_frame[sync_len] ^= 1
    payload_plus_crc = one_frame[sync_len:]
    assert verify_crc16(payload_plus_crc) is False
