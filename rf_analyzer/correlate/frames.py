"""Stages 13-14: Bitstream structure detection and header/payload identification."""
import numpy as np

def bits_to_bytes(bits):
    n = len(bits) - (len(bits) % 8)
    return np.packbits(bits[:n].reshape(-1, 8))

def crc16_ccitt(data_bytes, poly=0x1021, init=0xFFFF):
    """data_bytes: any iterable of byte values (e.g. a numpy uint8 array). Cast
    each value to a plain Python int first -- numpy's fixed-width uint8 dtype
    will raise an OverflowError on `b << 8` (which needs 16 bits of headroom)
    if you operate on it directly."""
    crc = init
    for b in data_bytes:
        b = int(b)
        crc ^= (b << 8)
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc

def find_repeating_frame_length(bits, min_len=32, max_len=2048):
    if len(bits) < max_len * 2:
        return None
        
    x = bits.astype(np.float32) * 2 - 1
    # Use FFT-based autocorrelation to avoid O(N^2) hang on long bitstreams
    # Pad to next power of 2 for speed
    n = len(x)
    n_fft = 1 << (n * 2 - 1).bit_length()
    X = np.fft.fft(x, n_fft)
    autocorr = np.fft.ifft(X * np.conj(X)).real
    
    # We only care about lags up to max_len
    autocorr = autocorr[:max_len+1] / n # Normalize
    
    best_lag = None
    best_score = 0
    
    for lag in range(min_len, min(max_len, n // 2)):
        score = autocorr[lag] * n / (n - lag)
        if score > best_score:
            best_score = score
            best_lag = lag
            
    if best_score == 0:
        return None
    baseline = np.mean(autocorr[min_len:max_len])
    confidence = float((best_score - baseline) / (autocorr[0] - baseline + 1e-9))
    return {"frame_length_bits": best_lag, "confidence": max(0.0, min(1.0, confidence))}

def find_header_payload_boundary(bits, frame_length, window_bits=64):
    if frame_length < window_bits:
        return None
    n_frames = len(bits) // frame_length
    if n_frames < 2:
        return None
    frames = bits[: n_frames * frame_length].reshape(n_frames, frame_length)
    col_variance = np.var(frames.astype(np.float32), axis=0)
    static_mask = col_variance < 0.02
    boundary = 0
    for i, is_static in enumerate(static_mask):
        if not is_static:
            boundary = i
            break
    else:
        boundary = frame_length
    return {"payload_offset_bits": int(boundary), "static_header_bits": int(boundary)}

def verify_crc16(payload_bits, crc_field_bits=16):
    payload_bytes = bits_to_bytes(payload_bits[:-crc_field_bits])
    received_crc_bits = payload_bits[-crc_field_bits:]
    received_crc = int("".join(map(str, received_crc_bits)), 2)
    computed = crc16_ccitt(payload_bytes)
    return computed == received_crc
