# RF Signal Analysis Workstation — Verified, Deployment-Ready Starter Code

This is the complete MVP codebase described in the accompanying project documents, **built, tested, and validated against realistic signal conditions** — not a plan awaiting execution. The full pipeline runs end-to-end and passes an automated 40-test suite, including a 30-trial randomized stress test.

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests/ -v                   # should show "40 passed"
streamlit run rf_analyzer/gui/app.py
```

## Quick start (Docker)

```bash
docker build -t rf-analyzer:1.0.0 .
docker run -d --name rf-analyzer -p 8501:8501 \
  -v "$(pwd)/data:/app/data" \
  rf-analyzer:1.0.0
# open http://localhost:8501
```

See **04_DEPLOYMENT_PLAN.md** for the full offline/air-gapped packaging procedure, rollback steps, and health checks.

## Production Deployment (Render & Docker)

WavQ is configured as a single Docker Web Service serving both the React frontend and FastAPI backend.

### 1. Local Development
```bash
# Backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
uvicorn backend.app:app --reload --port 8000

# Frontend (in separate terminal)
cd frontend_react
npm ci
npm run dev
```

### 2. Docker Build & Run
```bash
# Build production multi-stage container
docker build -t wavq .

# Run locally
docker run --rm -p 8000:8000 wavq
```

Once running:
- **Application UI**: `http://localhost:8000/`
- **Dashboard**: `http://localhost:8000/dashboard`
- **Workspace**: `http://localhost:8000/workspace`
- **Health Check**: `http://localhost:8000/api/health` -> `{"status": "ok"}`
- **Sample Signals**: `http://localhost:8000/api/samples`

### 3. Deploy to Render
1. Create a new service on [Render](https://render.com): **New +** -> **Web Service**.
2. Connect your GitHub repository (`WavQ`).
3. Select Environment / Runtime: **Docker**.
4. Leave **Build Command** and **Start Command** empty (Render automatically uses the `Dockerfile`).
5. Render automatically injects the `$PORT` environment variable.
6. Deployment URL pattern: `https://<your-app-name>.onrender.com`

*Note: Uploaded files and temporary analysis results under `./uploads` are ephemeral in cloud containers unless persistent disk storage is attached.*


## What's included

- `rf_analyzer/` — the full pipeline: ingest, spectral estimation, receive matched filtering + timing/carrier synchronization, modulation classification, demodulation, interleaver detection, FEC decoding, bitstream correlation, orchestration, GUI, and export.
- `tests/` — 40 pytest tests covering every module individually plus full end-to-end pipeline runs, including a 30-trial randomized stress test across modulation/carrier-offset/SNR combinations.
- `requirements.txt` — pinned, tested dependency versions.
- `Dockerfile`, `.streamlit/config.toml`, `.dockerignore` — ready-to-build offline deployment artifacts.

## Read this alongside

- **00_UNIFIED_BLUEPRINT.md** — what this system is, why it's architected this way, and an **Engineering Validation Journal** documenting eight real defects found and fixed by validating against realistically pulse-shaped signals instead of idealized ones — read this first, it explains design decisions that otherwise look arbitrary.
- **01_IMPLEMENTATION_PLAN.md** — step-by-step build guide for every file in `rf_analyzer/`, with design-history notes explaining *why* each module looks the way it does, not just what it does.
- **02_DEBUGGING_PLAN.md** — troubleshooting guide with dedicated entries for every real bug class found during this project's development, in case one is reintroduced.
- **03_TESTING_PLAN.md** — what every test proves and how to extend the suite.
- **04_DEPLOYMENT_PLAN.md** — offline/air-gapped packaging, verification, rollback, and monitoring.
- **05_EXECUTION_CHECKLIST_STATE_SUMMARY.md** — full lifecycle, master checklist (with accurate checked/unchecked status), and a project-state summary you can paste into a new chat to resume work.

## Current status

**MVP (MUST-HAVE) scope: complete and validated.** All 14 pipeline stages implemented, tested individually, tested end-to-end, and stress-tested across randomized conditions (30/30 trials passed). Read the blueprint's Engineering Validation Journal for the full account of what "validated" required — a first version passed every test while being fundamentally broken for realistic signals, and fixing that is most of this project's real engineering content.

**Known limitation, stated honestly:** all validation is against synthetic signals with known ground truth. This development environment could not reach public real-world capture archives (Zenodo, SatNOGS, Kaggle — confirmed blocked by network policy, not a missed step) to validate against actual recorded satellite telemetry. If you have access to real `.iq`/`.wav` captures, running them through this pipeline and extending the Engineering Validation Journal with whatever is found is the single highest-value next step.

**Not yet done:** manual UI walkthrough (needs a human), execution of the deployment steps against a real target machine, and the SHOULD-HAVE/NICE-TO-HAVE extensions listed in Part 6's checklist.
