# Future Work Backlog

The following research topics and algorithm improvements are deferred until after the v1.0 MVP stabilizes in production.

## 1. Algorithmic Enhancements
*   **Improved AMC**: Expand the Exact Mixture Likelihood model to handle higher-order QAM (64-QAM, 256-QAM) and APSK constellations.
*   **Better Blind Carrier Recovery**: The current Costas loop struggles with high residual CFO. Investigate feed-forward Viterbi & Viterbi or phase-unwrapping estimators for cleaner lock prior to decision-directed tracking.
*   **Stronger SNR Estimation**: The spectral SNR estimator is heavily biased by out-of-band noise floors. Implement data-aided or eigenvalue-based (e.g., M2M4) SNR estimators.

## 2. Demodulation & Decoding
*   **Advanced Forward Error Correction**: 
    *   Integrate Low-Density Parity-Check (LDPC) decoders.
    *   Integrate Turbo decoders.
    *   Add BCH block decoding.
*   **Advanced Interleaver Detection**: Implement matrix-rank or convolutional interleaver detection (current MVP only supports basic block interleaving up to width 256).

## 3. Data & Validation
*   **Additional RF Datasets**: Benchmark against larger over-the-air (OTA) datasets.
*   **OTA Validation**: Rigorous multi-path and fading validation (current MVP is primarily AWGN-calibrated).
