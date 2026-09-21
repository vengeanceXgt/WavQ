from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class SignalInfo(BaseModel):
    file_path: str = Field(..., description="Absolute path to the uploaded signal file")
    sample_rate: Optional[float] = Field(None, description="Sample rate in Hz")
    samples: Optional[int] = Field(None, description="Number of samples processed")
    duration_sec: Optional[float] = Field(None, description="Duration of the signal in seconds")

class AnalysisResult(BaseModel):
    status: str = Field(..., description="overall processing status: 'complete', 'partial', or 'failed'")
    input: SignalInfo
    signal: Dict[str, Any] = Field(default_factory=dict, description="Spectral and SNR results")
    synchronization: Dict[str, Any] = Field(default_factory=dict, description="CFO, symbol rate, timing, carrier lock status")
    modulation: Dict[str, Any] = Field(default_factory=dict, description="Modulation classification result")
    demodulation: Dict[str, Any] = Field(default_factory=dict, description="Demodulation outcome and bit count")
    frame: Dict[str, Any] = Field(default_factory=dict, description="Frame detection details")
    fec: Dict[str, Any] = Field(default_factory=dict, description="FEC identification and decoding result")
    evidence: List[Any] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    processing_timestamp: Optional[float] = Field(None, description="Unix timestamp when processing completed")
    version: Optional[str] = Field(None, description="Version identifier of the backend pipeline")

# Unified response model that includes the unique identifier for the uploaded signal
class SignalResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the uploaded signal")
    status: str = Field(..., description="Processing status returned by the pipeline")
    result: AnalysisResult
