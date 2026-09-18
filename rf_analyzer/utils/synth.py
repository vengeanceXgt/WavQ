"""Synthetic RF signal generator for testing. Produces known-answer signals.

IMPORTANT HISTORY: an earlier version of this generator used rectangular
(unshaped) NRZ pulses -- literally repeating each symbol value across `sps`
samples with instantaneous transitions. That is NOT what any real transmitter
does (it would waste infinite bandwidth), and it turned out to be more than a
cosmetic simplification: it's a mathematically DEGENERATE case for the Gardner
timing-error detector used in the synchronization stage (Gardner relies on the
inter-symbol-interference-induced amplitude dip at the half-symbol point to
sense timing error; with instantaneous, sample-aligned transitions, that
information doesn't meaningfully exist). This generator now applies realistic
root-raised-cosine (RRC) pulse shaping by default, which is both a more
faithful stand-in for real captured signals and the shape needed for the
synchronization stage to work correctly at all. See the sync module's design
notes for the full story of how this was found.
"""
import numpy as np

def random_bits(n_bits, seed=None):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)

CONSTELLATIONS = {
    "bpsk": np.array([-1, 1], dtype=complex),
    "qpsk": np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex) / np.sqrt(2),
    "16qam": np.array([complex(i, q) for i in (-3,-1,1,3) for q in (-3,-1,1,3)]) / np.sqrt(10),
}

def rrc_filter(beta, span, sps):
    """Root-raised-cosine FIR filter, unit-energy normalized."""
    N = span * sps
    t = np.arange(-N, N + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta != 0 and abs(abs(4 * beta * ti) - 1.0) < 1e-8:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*beta)))
        else:
            num = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1 - (4*beta*ti)**2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))

def modulate(bits, scheme, sps=8, rrc_beta=0.35, rrc_span=6):
    """Map bits to a complex baseband waveform, RRC-pulse-shaped (realistic).
    sps = samples per symbol."""
    if scheme == "2fsk":
        symbols = np.where(bits == 1, 1.0, -1.0)
        phase = np.cumsum(np.repeat(symbols, sps)) * (np.pi / sps)
        return np.exp(1j * phase)  # CPFSK is already smooth/continuous-phase by construction

    const = CONSTELLATIONS[scheme]
    bits_per_sym = int(np.log2(len(const)))
    padded = bits[: len(bits) - (len(bits) % bits_per_sym)]
    groups = padded.reshape(-1, bits_per_sym)
    idx = groups.dot(1 << np.arange(bits_per_sym - 1, -1, -1))
    symbols = const[idx]

    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    h = rrc_filter(rrc_beta, rrc_span, sps)
    shaped = np.convolve(upsampled, h, mode="full")
    # trim the filter's startup/settling transient (rrc_span symbols on each side)
    trim = rrc_span * sps
    return shaped[trim: trim + len(symbols) * sps]

def add_impairments(iq, cfo_hz=0.0, sample_rate=1e6, snr_db=20.0, seed=None):
    rng = np.random.default_rng(seed)
    n = len(iq)
    t = np.arange(n) / sample_rate
    iq = iq * np.exp(1j * 2 * np.pi * cfo_hz * t)  # carrier frequency offset
    sig_power = np.mean(np.abs(iq) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(scale=np.sqrt(noise_power / 2), size=n) + \
            1j * rng.normal(scale=np.sqrt(noise_power / 2), size=n)
    return iq + noise

def generate_test_signal(scheme="qpsk", n_bits=2000, sps=8, sample_rate=1e6,
                          cfo_hz=1500.0, snr_db=15.0, seed=42, rrc_beta=0.35):
    """Returns (iq_samples, ground_truth_dict). RRC-pulse-shaped by default
    (realistic); pass rrc_beta=None for the old unshaped/rectangular behavior
    (kept only for comparison/debugging -- do not use for real validation, see
    the module docstring)."""
    bits = random_bits(n_bits, seed=seed)
    if scheme != "2fsk" and rrc_beta is None:
        const = CONSTELLATIONS[scheme]
        bits_per_sym = int(np.log2(len(const)))
        padded = bits[: len(bits) - (len(bits) % bits_per_sym)]
        groups = padded.reshape(-1, bits_per_sym)
        idx = groups.dot(1 << np.arange(bits_per_sym - 1, -1, -1))
        baseband = np.repeat(const[idx], sps)
    else:
        baseband = modulate(bits, scheme, sps=sps, rrc_beta=rrc_beta or 0.35)
    iq = add_impairments(baseband, cfo_hz=cfo_hz, sample_rate=sample_rate,
                          snr_db=snr_db, seed=seed)
    truth = {
        "modulation": scheme, "n_bits": n_bits, "sps": sps,
        "sample_rate": sample_rate, "cfo_hz": cfo_hz, "snr_db": snr_db,
        "bits": bits, "rrc_beta": rrc_beta,
    }
    return iq.astype(np.complex64), truth
