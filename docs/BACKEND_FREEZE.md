# NTROv5 Backend Freeze

## Architecture
The backend is now frozen for NTROv5 MVP. The core execution engine is `rf_analyzer.orchestrator.pipeline.analyze_signal`, which ingests a generic signal (IQ/WAV) and routes it through sequential stages (preprocessing, spectral, synchronization, modulation, demodulation, FEC), catching stage-level exceptions to gracefully degrade functionality and produce a `partial` status rather than a crash.

## API Endpoints
* `GET /health` : Returns `{ "status": "ok", "version": "5.0.0-mvp" }`
* `POST /analyze` : Accepts `multipart/form-data` with a `file` field. Returns a standard JSON schema containing measurement and classification data.

## Result Schema
The JSON result schema provides standardized fields:
* `status`: `complete`, `partial`, `failed`, `unsupported`
* `limitations`: List of string warnings when downstream stages fail
* `input`: Details of ingested format and length
* `signal`: Spectral estimations (bandwidth, CFO, SNR)
* `synchronization`: Symbol rate and Costas recovery metrics
* `modulation`: Classification (`label`, `ambiguous`, `evidence`)
* `demodulation`: Demodulation statistics, EVM, mapped bits
* `frame`: Synchronization boundaries and headers
* `fec`: Interleaving matrices and decoded payloads

## Supported Capabilities

### Input Formats
* Raw IQ (Float32 Complex)
* WAV (Real or Complex Audio-Band)
* SigMF (.sigmf-meta)

### Analysis
* Bandwidth
* CFO (Nonlinear Spectral)
* Burst Detection (Energy Envelope)
* SNR (Spectral noise floor)

### Synchronization
* Symbol Rate (Cyclic Spectrum)
* Gardner Timing Recovery
* Costas Carrier Phase Recovery

### Modulation & Demodulation
* BPSK
* QPSK
* 16QAM
* 2-FSK (Classification heuristic only)

### FEC
* Viterbi (K=7, Rate 1/2)
* Reed-Solomon
* Block Interleaving

## Known Limitations / Deferred
* LDPC, Turbo, BCH are explicitly deferred.
* 64QAM, 256QAM, APSK, CPM are explicitly deferred.
* Costas loop cycle slipping requires higher SNR than the exact ML classifier requires to correctly identify QPSK/16QAM, leading to valid classifications that yield `partial` processing statuses because phase track is lost prior to demodulation.
* Ambiguous modulations (e.g. unknown mapping variants) are flagged with `ambiguous: true` and will explicitly halt the demodulation pipeline.
