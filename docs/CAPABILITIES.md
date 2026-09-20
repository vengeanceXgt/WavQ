# RF Analyzer Capabilities

The following capabilities represent strictly tested, implemented, and empirically supported functionality in the MVP version.

### Input
* IQ (Raw Complex Float32)
* WAV (Real or Complex Audio-Band)
* SigMF Data (Paired with `.sigmf-meta`)

### Signal analysis
* Occupied Bandwidth Estimation
* Carrier Frequency Offset (CFO) Estimation (Nonlinear Spectral)
* Burst Detection (Energy-based Envelope)
* SNR / C/N Estimation (Spectral noise floor projection)

### Synchronization
* Symbol-rate Estimation (Cyclic Spectrum)
* Gardner Timing Recovery (Blind)
* Costas Carrier Phase Recovery (Decision-Directed QPSK/16QAM)

### Modulation
* BPSK (Exact Mixture Likelihood)
* QPSK (Exact Mixture Likelihood)
* 16QAM (Exact Mixture Likelihood)
* 2-FSK (Envelope / FM Discriminator heuristics)

### Demodulation
* BPSK
* QPSK
* 16QAM

### FEC & Interleaving
* Viterbi (K=7, Rate 1/2 Convolutional)
* Reed-Solomon
* Block Interleaving (Matrix Widths up to 256)

### Not Supported (Future Work)
* LDPC
* Turbo
* BCH
* 64QAM
* 256QAM
* APSK
* CPM
* Feed-forward Phase-unwrapping Carrier Recovery
