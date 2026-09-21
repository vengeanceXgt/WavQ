import axios from 'axios';

export interface SampleSignal {
  name: string;
  size_bytes: number;
  type: string;
  label: string;
  description: string;
}

export interface SignalAnalysisResult {
  id: string;
  status: string;
  result: {
    status: string;
    input?: {
      file_path?: string;
      sample_rate?: number;
      samples?: number;
      duration_sec?: number;
    };
    signal?: {
      waveform?: number[];
      spectrogram?: number[][];
      psd?: { freqs: number[]; powers: number[] };
      snr_db?: number;
      bandwidth?: number;
      center_freq?: number;
      sample_rate?: number;
      samples?: number;
    };
    synchronization?: {
      cfo_hz?: number;
      symbol_rate?: number;
      timing_status?: string;
      carrier_phase?: number[];
    };
    modulation?: {
      label?: string;
      ambiguous?: boolean | number;
      candidates?: Array<{ label: string; score: number; sigma_sq?: number }>;
      evidence?: {
        method?: string;
        margin?: number;
        best_score?: number;
        second_score?: number;
        cumulants?: Record<string, any>;
      };
    };
    demodulation?: {
      status?: string;
      bit_count?: number;
      constellation?: { i: number[]; q: number[] };
      bits_preview?: string;
      hex_preview?: string;
    };
    frame?: {
      status?: string;
      frame_length?: number;
      header_boundary?: number;
      payload_length?: number;
      syncword?: string;
    };
    fec?: {
      status?: string;
      code_rate?: string;
      parity_status?: string;
      ber_estimate?: number;
    };
    evidence?: string[];
    limitations?: string[];
  };
}

export const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
});

export const checkHealth = async (): Promise<boolean> => {
  try {
    const res = await api.get('/health');
    return res.data?.status === 'ok';
  } catch {
    return false;
  }
};

export const fetchSamples = async (): Promise<SampleSignal[]> => {
  try {
    const res = await api.get('/samples');
    return res.data?.samples || [];
  } catch (err) {
    console.warn('Failed to fetch samples:', err);
    return [];
  }
};

export const analyzeSampleSignal = async (sampleName: string): Promise<SignalAnalysisResult> => {
  const res = await api.post(`/samples/${encodeURIComponent(sampleName)}/analyze`);
  return res.data;
};

export const uploadSignal = async (file: File): Promise<SignalAnalysisResult> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/signals', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const getSignalResult = async (id: string): Promise<SignalAnalysisResult> => {
  const response = await api.get(`/signals/${id}`);
  return response.data;
};
