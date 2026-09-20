class FrequencyResult:
    def __init__(self, relative_offset_hz, absolute_rf_hz, source, reason=""):
        self.relative_offset_hz = relative_offset_hz
        self.absolute_rf_hz = absolute_rf_hz
        self.source = source
        self.reason = reason

def resolve_frequencies(relative_offset_hz, sigmf_metadata=None, manual_rf_hz=None):
    """
    Distinguishes relative baseband offset vs SigMF absolute RF frequency.
    """
    if manual_rf_hz is not None:
        abs_rf = manual_rf_hz + relative_offset_hz
        return FrequencyResult(relative_offset_hz, float(abs_rf), "user", "")
        
    if sigmf_metadata and "captures" in sigmf_metadata and isinstance(sigmf_metadata["captures"], list) and len(sigmf_metadata["captures"]) > 0:
        try:
            capture_freq = sigmf_metadata["captures"][0]["core:frequency"]
            abs_rf = capture_freq + relative_offset_hz
            return FrequencyResult(relative_offset_hz, float(abs_rf), "metadata", "")
        except KeyError:
            pass
            
    return FrequencyResult(relative_offset_hz, None, "unavailable", "RF reference unavailable")
