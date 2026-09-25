# backend/app.py
import os
import shutil
import uuid
from pathlib import Path
import json

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from .schemas import SignalResponse

# Import the existing analysis pipeline
from rf_analyzer.orchestrator.pipeline import analyze_signal

app = FastAPI(title="NTROv5 Signal Analysis API")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded temporary files
UPLOAD_ROOT = Path("./uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/signals", response_model=SignalResponse)
async def upload_signal(file: UploadFile = File(...)):
    # Validate extension safely
    filename = file.filename or "uploaded.iq"
    allowed_ext = {".iq", ".wav", ".dat", ".bin", ".sigmf-data"}
    ext = Path(filename).suffix.lower()
    if not ext or ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Create a unique temporary directory for this upload
    signal_id = str(uuid.uuid4())
    dest_dir = UPLOAD_ROOT / signal_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"input{ext}"

    # Write the file to disk
    with dest_path.open("wb") as out_file:
        while content := await file.read(1024 * 1024):
            out_file.write(content)

    # Run the analysis pipeline
    try:
        result = analyze_signal(str(dest_path))
    except Exception as exc:
        # Cleanup and propagate error
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # Build response model
    response = SignalResponse(id=signal_id, status=result.get("status", "unknown"), result=result)
    # Store result JSON for later retrieval using Pydantic v2 model_dump_json()
    result_path = dest_dir / "result.json"
    result_json = response.model_dump_json() if hasattr(response, "model_dump_json") else response.json()
    result_path.write_text(result_json)

    return response

SAMPLES_ROOT = Path("./data/samples")

@app.get("/api/samples")
def list_samples():
    samples = []
    if SAMPLES_ROOT.exists():
        for p in SAMPLES_ROOT.iterdir():
            if p.is_file() and p.suffix.lower() in {".iq", ".wav", ".dat", ".bin"}:
                samples.append({
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "type": p.suffix.lower().lstrip("."),
                    "label": "BPSK Smoke Test (2 MHz)" if "bpsk" in p.name.lower() else ("QPSK High SNR (8 MHz)" if "qpsk" in p.name.lower() else p.stem),
                    "description": "Standard binary phase-shift keyed signal with 10k symbols" if "bpsk" in p.name.lower() else "Quadrature phase-shift keyed signal with carrier offset"
                })
    return {"samples": samples}

@app.post("/api/samples/{sample_name}/analyze", response_model=SignalResponse)
def analyze_sample(sample_name: str):
    # Sanitize filename
    safe_name = Path(sample_name).name
    sample_path = SAMPLES_ROOT / safe_name
    if not sample_path.exists() or not sample_path.is_file():
        raise HTTPException(status_code=404, detail="Sample not found")
    
    signal_id = f"sample-{sample_path.stem}"
    dest_dir = UPLOAD_ROOT / signal_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = analyze_signal(str(sample_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
        
    response = SignalResponse(id=signal_id, status=result.get("status", "unknown"), result=result)
    result_path = dest_dir / "result.json"
    result_json = response.model_dump_json() if hasattr(response, "model_dump_json") else response.json()
    result_path.write_text(result_json)
    return response

@app.get("/api/signals/{signal_id}")
def get_signal_result(signal_id: str):
    result_path = UPLOAD_ROOT / signal_id / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return JSONResponse(content=json.loads(result_path.read_text()))

# Serve static frontend files (React production build in frontend_react/dist)
FRONTEND_DIST = Path("frontend_react/dist").resolve()

if FRONTEND_DIST.exists() and (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="static_assets")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Never intercept API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")

    if not FRONTEND_DIST.exists():
        raise HTTPException(status_code=404, detail="Frontend dist directory not found")

    # Sanitize requested path to prevent path traversal
    requested_path = (FRONTEND_DIST / full_path).resolve()
    try:
        requested_path.relative_to(FRONTEND_DIST)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    # If file exists and is a file, serve it directly
    if requested_path.is_file():
        return FileResponse(requested_path)

    # Fallback to index.html for client-side React routes (e.g. /dashboard, /workspace)
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    raise HTTPException(status_code=404, detail="Frontend index.html not found")
