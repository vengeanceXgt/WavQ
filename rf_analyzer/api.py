import tempfile
import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rf_analyzer.orchestrator.pipeline import analyze_signal

app = FastAPI(title="RF Analyzer API", version="1.0.0")

class HealthCheck(BaseModel):
    status: str

@app.get("/health", response_model=HealthCheck)
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".iq"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = analyze_signal(tmp_path)
    finally:
        os.remove(tmp_path)

    return {
        "analysis_id": file.filename,
        "status": result["status"],
        "summary": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
