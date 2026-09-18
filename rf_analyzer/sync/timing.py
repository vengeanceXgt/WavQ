"""Stage 8 (part 1): Gardner timing-error-detector symbol timing recovery.

DESIGN HISTORY (read before modifying): this module went through FOUR
distinct bug fixes, each invisible until tested under the right conditions:

  v1 (unbounded integrator): worked on short signals, slipped a full sample
     on long streams. Fixed by removing integration entirely.

  v2 (proportional-only, wrong output tap "mid"): passed every test because
     every test used unshaped rectangular pulses, where Gardner's error
     detector is mathematically degenerate. Failed badly on realistic
     RRC-shaped signals.

  v3 (proportional-integral, leaky bounded integrator, "early" tap): fixed
     v2's problem by validating against RRC-shaped signals -- but only at
     one specific, arbitrary signal amplitude.

  v4 (THIS VERSION -- power-normalized error term): found by testing the same
     v3 code through the full pipeline (which applies a power-normalization
     step upstream, in ingest.normalize()) versus testing it directly on
     un-normalized synthetic signals. Results differed substantially --
     tracked down to the Gardner error term's absolute magnitude scaling with
     the SQUARE of input amplitude (it's a product of two amplitude-scale
     quantities), while alpha/beta were fixed constants. A signal scaled up
     or down therefore changed the loop's EFFECTIVE gain/bandwidth, degrading
     tracking quality in an amplitude-dependent way -- exactly the kind of
     thing a proper AGC-invariant timing detector must not do. The fix
     normalizes the error term by the signal's own estimated power, making
     the loop's behavior independent of absolute input amplitude, as it
     should be regardless of what normalization (if any) happened upstream.

The lesson generalizes: **validate DSP code across the range of realistic
conditions it will actually see in the full pipeline, not just in isolation
at one convenient amplitude/shape/SNR** -- three of these four bugs were
completely invisible in isolated, idealized unit tests and only surfaced once
tested end-to-end or under more realistic conditions.
"""
import numpy as np

def gardner_timing_recovery(iq, sps, alpha=0.02, beta=0.0005, leak=0.995):
    """Returns one complex symbol per detected symbol period.
    alpha: proportional gain. beta: integral gain (settles to nonzero offsets).
    leak: integrator decay per step (<1.0), bounds long-term drift.
    The error term is normalized by the signal's estimated average power so
    the loop's effective gain is independent of the input's absolute
    amplitude (see the design note above)."""
    power_est = np.mean(np.abs(iq) ** 2) + 1e-12
    mu_int = 0.0
    out = []
    i = 0.0
    n = len(iq)
    while i + 2 * sps < n:
        idx = int(round(i))
        early = iq[idx]
        mid = iq[idx + sps // 2]
        late = iq[idx + sps]
        error = np.real(np.conj(mid) * (late - early)) / power_est
        mu_int = np.clip(leak * mu_int + beta * error, -1.5, 1.5)
        correction = np.clip(alpha * error + mu_int, -1.5, 1.5)
        out.append(early)
        i += sps + correction
    return np.array(out, dtype=np.complex64)
