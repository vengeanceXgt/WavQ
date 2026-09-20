# NTROv5 Migration Guide

## Preserved
- `rf_analyzer/amc/cnn_model.py`: The original NTROv4 CNN architecture and RadioML model weights are preserved untouched for fallback use.
- `rf_analyzer/sync/`: Gardner timing recovery, Costas loop, matched filtering algorithms remain structurally intact.
- `rf_analyzer/spectral/`: Cyclostationary symbol rate estimator, cyclic spectral density, and nonlinear CFO estimators.

## Added
- **Exact ML AMC**: `rf_analyzer/amc/exact_ml.py` replaces the raw CNN confidence with a strictly calibrated Exact Mixture Likelihood estimator fused with higher-order cumulants.
- **Pipeline Orchestrator**: `rf_analyzer/orchestrator/pipeline.py` introduces a robust schema-driven execution framework that catches per-stage failures and prevents crashing.
- **GUI & API**: `rf_analyzer/gui/app.py` and `rf_analyzer/api.py` added for Streamlit and FastAPI deployment.
- **Configuration**: `rf_analyzer/config.py` isolates DSP constraints.
- **Docker**: Containerization definitions (`Dockerfile`, `docker-compose.yml`, `start.sh`) for rapid deployment.

## Modified
- `rf_analyzer/amc/classifier.py`: Switched default classification from CNN to the new Exact ML algorithm. Returns standard schema.
- `tests/test_pipeline_integration.py`: Migrated away from deprecated `run_pipeline` `ok`/`error` array schema to the new unified JSON schema `analyze_signal`.
- `tests/test_radioml2018.py`: Removed deprecated `use_cnn_amc` arguments that were conflicting with the new `run_pipeline_on_iq` signature.

## Removed
- N/A - No production files were permanently removed. Only deprecated schema arguments from tests.

## Deprecated
- `run_pipeline` in `rf_analyzer/orchestrator/pipeline.py` is deprecated in favor of `analyze_signal`.

## Known behavior changes
- AMC now returns explicit "ambiguous" flags rather than arbitrary percentage confidences.
- Unsuccessful demodulation returns `status: partial` rather than crashing the pipeline.
