import numpy as np

class BurstSegment:
    def __init__(self, start_sample, end_sample, sample_rate, peak_power, mean_power, noise_power, threshold_on, threshold_off):
        self.start_sample = start_sample
        self.end_sample = end_sample
        self.start_time_s = start_sample / sample_rate
        self.end_time_s = end_sample / sample_rate
        self.duration_s = (end_sample - start_sample) / sample_rate
        self.peak_power = peak_power
        self.mean_power = mean_power
        self.noise_power = noise_power
        self.threshold_on = threshold_on
        self.threshold_off = threshold_off

def detect_bursts(iq, sample_rate, min_duration_s=0.001, min_gap_s=0.001, smooth_window=11, 
                  threshold_on_db=10.0, threshold_off_db=3.0):
    """
    Detects bursts in a complex IQ signal using an adaptive energy-based hysteresis state machine.
    """
    if len(iq) == 0:
        return []
        
    # Handle NaN/Inf safely
    iq = np.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Instantaneous power
    power = np.real(iq)**2 + np.imag(iq)**2
    
    # Optional smoothing
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        power_smooth = np.convolve(power, kernel, mode='same')
    else:
        power_smooth = power
        
    # Robust noise-floor estimate (use a low percentile to support high-duty-cycle signals)
    noise_power = np.percentile(power_smooth, 1)
    if noise_power <= 0:
        noise_power = 1e-12 # avoid log of zero or negative
        
    # Adaptive thresholds
    threshold_on = noise_power * (10 ** (threshold_on_db / 10.0))
    threshold_off = noise_power * (10 ** (threshold_off_db / 10.0))
    
    # Hysteresis state machine
    state_on = False
    bursts_raw = []
    current_start = 0
    
    for i, p in enumerate(power_smooth):
        if not state_on:
            if p >= threshold_on:
                state_on = True
                current_start = i
        else:
            if p <= threshold_off:
                state_on = False
                bursts_raw.append((current_start, i))
                
    # Handle burst spanning to the end of the file
    if state_on:
        bursts_raw.append((current_start, len(power_smooth)))
        
    # Merge short gaps
    bursts_merged = []
    min_gap_samples = int(min_gap_s * sample_rate)
    
    if len(bursts_raw) > 0:
        current_burst = bursts_raw[0]
        for next_burst in bursts_raw[1:]:
            gap = next_burst[0] - current_burst[1]
            if gap < min_gap_samples:
                # Merge
                current_burst = (current_burst[0], next_burst[1])
            else:
                bursts_merged.append(current_burst)
                current_burst = next_burst
        bursts_merged.append(current_burst)
        
    # Minimum duration filtering and segment creation
    min_duration_samples = int(min_duration_s * sample_rate)
    final_bursts = []
    
    for start, end in bursts_merged:
        if (end - start) >= min_duration_samples:
            burst_power = power[start:end]
            peak = float(np.max(burst_power))
            mean = float(np.mean(burst_power))
            final_bursts.append(BurstSegment(
                start_sample=start,
                end_sample=end,
                sample_rate=sample_rate,
                peak_power=peak,
                mean_power=mean,
                noise_power=float(noise_power),
                threshold_on=float(threshold_on),
                threshold_off=float(threshold_off)
            ))
            
    return final_bursts
