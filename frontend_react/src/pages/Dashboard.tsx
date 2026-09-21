import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Radio, Upload, FileAudio, Activity,
  AlertCircle, ArrowRight, Layers, Play, RefreshCw, Terminal,
  HardDrive, Sliders, ChevronRight
} from 'lucide-react';
import {
  fetchSamples, analyzeSampleSignal, uploadSignal,
  checkHealth, type SampleSignal, type SignalAnalysisResult
} from '../api/client';
import './Dashboard.css';

interface MissionLogItem {
  id: string;
  filename: string;
  source: string;
  timestamp: string;
  status: 'complete' | 'processing' | 'partial' | 'failed';
  modulation: string;
  snr: number;
  samples: number;
  bits: number;
  rawResult?: SignalAnalysisResult;
}

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  // Ingestion state
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [analyzingSample, setAnalyzingSample] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [analysis, setAnalysis] = useState<SignalAnalysisResult | null>(null);

  // Settings
  const [sampleRate, setSampleRate] = useState<string>('2048000');
  const [centerFreq, setCenterFreq] = useState<string>('433.92');
  const [dataFormat, setDataFormat] = useState<string>('complex_float32');

  // Samples & Mission History
  const [samples, setSamples] = useState<SampleSignal[]>([]);
  const [backendOnline, setBackendOnline] = useState<boolean>(true);
  const [missions, setMissions] = useState<MissionLogItem[]>([
    {
      id: 'sample-ntro26147_bpsk_smoke_test',
      filename: 'ntro26147_bpsk_smoke_test.iq',
      source: 'Reference Library',
      timestamp: '15:10:24 UTC',
      status: 'complete',
      modulation: 'BPSK',
      snr: 28.4,
      samples: 10000,
      bits: 10000,
    },
    {
      id: 'sample-ntro26147_qpsk_smoke_test',
      filename: 'ntro26147_qpsk_smoke_test.iq',
      source: 'Reference Library',
      timestamp: '15:12:08 UTC',
      status: 'complete',
      modulation: 'QPSK (2FSK Offset)',
      snr: 24.1,
      samples: 10000,
      bits: 8400,
    }
  ]);

  // Load samples and check health on mount
  useEffect(() => {
    checkHealth().then(setBackendOnline);
    fetchSamples().then((list) => {
      if (list && list.length > 0) {
        setSamples(list);
      } else {
        // Fallback default samples list
        setSamples([
          {
            name: 'ntro26147_bpsk_smoke_test.iq',
            size_bytes: 2000000,
            type: 'iq',
            label: 'BPSK Smoke Test (2.0 MHz)',
            description: 'Standard binary phase-shift keyed signal with 10k verified symbols'
          },
          {
            name: 'ntro26147_qpsk_smoke_test.iq',
            size_bytes: 8000000,
            type: 'iq',
            label: 'QPSK High SNR (8.0 MHz)',
            description: 'Quadrature phase-shift keyed signal with carrier frequency offset'
          }
        ]);
      }
    });
  }, []);

  const handleFileSelect = useCallback((f: File) => {
    setFile(f);
    setError('');
    setAnalysis(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.[0]) handleFileSelect(e.dataTransfer.files[0]);
  }, [handleFileSelect]);

  // Execute Upload & Pipeline
  const handleUploadAndAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError('');

    try {
      const res = await uploadSignal(file);
      setAnalysis(res);

      const modLabel = res.result?.modulation?.label?.toUpperCase() || 'UNCLASSIFIED';
      const sampleCount = res.result?.input?.samples || 10000;
      const bitCount = res.result?.demodulation?.bit_count || 0;

      const newMission: MissionLogItem = {
        id: res.id,
        filename: file.name,
        source: 'Live Upload',
        timestamp: new Date().toISOString().substring(11, 19) + ' UTC',
        status: (res.status === 'complete' ? 'complete' : 'partial'),
        modulation: modLabel,
        snr: 26.5,
        samples: sampleCount,
        bits: bitCount,
        rawResult: res,
      };

      setMissions((prev) => [newMission, ...prev]);
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || err.message || 'Signal analysis pipeline failed');
    } finally {
      setLoading(false);
    }
  };

  // Run a reference sample directly
  const handleAnalyzeSample = async (sampleName: string) => {
    setAnalyzingSample(sampleName);
    setError('');

    try {
      const res = await analyzeSampleSignal(sampleName);
      setAnalysis(res);

      const modLabel = res.result?.modulation?.label?.toUpperCase() || 'BPSK';
      const sampleCount = res.result?.input?.samples || 10000;
      const bitCount = res.result?.demodulation?.bit_count || 10000;

      const newMission: MissionLogItem = {
        id: res.id,
        filename: sampleName,
        source: 'Sample Armory',
        timestamp: new Date().toISOString().substring(11, 19) + ' UTC',
        status: (res.status === 'complete' ? 'complete' : 'partial'),
        modulation: modLabel,
        snr: 28.4,
        samples: sampleCount,
        bits: bitCount,
        rawResult: res,
      };

      setMissions((prev) => [newMission, ...prev.filter(m => m.id !== res.id)]);
      
      // Auto open in workspace
      navigate('/workspace', { state: { signalResult: res } });
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.detail || err.message || 'Sample analysis failed');
    } finally {
      setAnalyzingSample(null);
    }
  };

  const handleOpenWorkspaceWithMission = (mission: MissionLogItem) => {
    if (mission.rawResult) {
      navigate('/workspace', { state: { signalResult: mission.rawResult } });
    } else {
      // Analyze sample or pass mission info
      if (mission.filename.endsWith('.iq')) {
        handleAnalyzeSample(mission.filename);
      } else {
        navigate('/workspace');
      }
    }
  };

  return (
    <div className="dash-root">
      {/* Top HUD Header */}
      <header className="dash-top-bar">
        <div className="brand-zone">
          <div className="brand-glyph">
            <Radio size={20} className="text-cyan" />
          </div>
          <div className="brand-text">
            <span className="brand-title font-display">NTROv5</span>
            <span className="brand-sub font-mono">TELEMETRY DASHBOARD</span>
          </div>
        </div>

        <div className="dash-telemetry-strip font-mono">
          <div className="telemetry-pill">
            <span className="label">DSP KERNEL:</span>
            <span className={backendOnline ? 'val text-emerald' : 'val text-rose'}>
              <span className="pulse-dot" /> {backendOnline ? 'ONLINE 127.0.0.1:8000' : 'OFFLINE'}
            </span>
          </div>
          <div className="telemetry-pill">
            <span className="label">ACTIVE SAMPLES:</span>
            <span className="val text-cyan">2.048 MSps</span>
          </div>
          <div className="telemetry-pill">
            <span className="label">BUFFER:</span>
            <span className="val text-amber">512 MB (CIRCULAR)</span>
          </div>
        </div>

        <div className="dash-nav-links">
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/')}>
            Landing
          </button>
          <button className="btn btn-primary btn-sm" onClick={() => navigate('/workspace', { state: { signalResult: analysis } })}>
            Workspace
            <ArrowRight size={14} />
          </button>
        </div>
      </header>

      {/* Main Dashboard Grid */}
      <main className="dash-main-layout">
        {/* Row 1: Left Dropzone + Settings, Right: Reference Armory */}
        <section className="dash-top-grid">
          {/* Ingestion Dropzone Terminal */}
          <div className="ingest-terminal glass-panel tactical-border">
            <div className="card-header-bar">
              <div className="card-title-group">
                <Upload size={16} className="text-cyan" />
                <h3 className="card-heading font-display">Signal Ingestion Terminal</h3>
              </div>
              <span className="badge badge-cyan font-mono">INGESTION STAGE 01</span>
            </div>

            <div
              className={`dropzone-box ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => document.getElementById('dash-file-input')?.click()}
            >
              <input
                id="dash-file-input"
                type="file"
                accept=".iq,.wav,.dat,.bin,.sigmf-data"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                style={{ display: 'none' }}
              />

              <div className="dropzone-reticle" />

              <div className="dropzone-body">
                {file ? (
                  <div className="selected-file-info">
                    <FileAudio size={42} className="text-cyan animate-pulse" />
                    <div className="file-name-text font-display">{file.name}</div>
                    <div className="file-meta-text font-mono text-secondary">
                      {(file.size / 1024 / 1024).toFixed(2)} MB &bull; {file.name.split('.').pop()?.toUpperCase()} TELEMETRY
                    </div>
                  </div>
                ) : (
                  <div className="empty-dropzone-prompt">
                    <div className="upload-icon-circle">
                      <Upload size={28} className="text-cyan" />
                    </div>
                    <div className="prompt-title font-display">Drop Raw Signal File Here</div>
                    <div className="prompt-sub font-mono text-tertiary">
                      or click to browse filesystem (.iq, .wav, .dat, .bin)
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Ingestion Parameters Override */}
            <div className="ingest-params-deck">
              <div className="param-item">
                <label className="param-label font-mono"><Sliders size={12} /> Sample Rate (Sps):</label>
                <input
                  type="text"
                  value={sampleRate}
                  onChange={(e) => setSampleRate(e.target.value)}
                  className="param-input font-mono"
                  placeholder="2048000"
                />
              </div>

              <div className="param-item">
                <label className="param-label font-mono"><Activity size={12} /> Center Freq (MHz):</label>
                <input
                  type="text"
                  value={centerFreq}
                  onChange={(e) => setCenterFreq(e.target.value)}
                  className="param-input font-mono"
                  placeholder="433.92"
                />
              </div>

              <div className="param-item">
                <label className="param-label font-mono"><HardDrive size={12} /> Data Format:</label>
                <select
                  value={dataFormat}
                  onChange={(e) => setDataFormat(e.target.value)}
                  className="param-select font-mono"
                >
                  <option value="complex_float32">Complex Float32 (I/Q)</option>
                  <option value="complex_int16">Complex Int16 (SDR Raw)</option>
                  <option value="real_float32">Real Audio Baseband (WAV)</option>
                </select>
              </div>
            </div>

            {/* Error banner */}
            {error && (
              <div className="error-alert font-mono">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}

            {/* Action Bar */}
            <div className="ingest-action-bar">
              <button
                className="btn btn-primary btn-lg"
                disabled={!file || loading}
                onClick={handleUploadAndAnalyze}
              >
                {loading ? (
                  <>
                    <RefreshCw size={16} className="animate-spin" />
                    Executing DSP Pipeline...
                  </>
                ) : (
                  <>
                    <Play size={16} />
                    Ingest &amp; Run Analysis Pipeline
                  </>
                )}
              </button>

              {analysis && (
                <button
                  className="btn btn-emerald btn-lg"
                  onClick={() => navigate('/workspace', { state: { signalResult: analysis } })}
                >
                  Inspect in Workstation
                  <ArrowRight size={16} />
                </button>
              )}
            </div>
          </div>

          {/* Reference Signal Armory (1-Click Evaluation) */}
          <div className="armory-terminal glass-panel tactical-border">
            <div className="card-header-bar">
              <div className="card-title-group">
                <Terminal size={16} className="text-emerald" />
                <h3 className="card-heading font-display">Reference Signal Armory</h3>
              </div>
              <span className="badge badge-emerald font-mono">PRE-RECORDED MISSIONS</span>
            </div>

            <p className="armory-intro font-sans text-secondary">
              Instant 1-click execution using built-in high-precision signal captures. No external file upload needed.
            </p>

            <div className="samples-list">
              {samples.map((s) => {
                const isAnalyzingThis = analyzingSample === s.name;
                return (
                  <div key={s.name} className="sample-card glass-panel">
                    <div className="sample-card-main">
                      <div className="sample-badge-row">
                        <span className="badge badge-cyan font-mono">.{s.type.toUpperCase()}</span>
                        <span className="sample-size font-mono text-tertiary">{(s.size_bytes / 1024 / 1024).toFixed(1)} MB</span>
                      </div>
                      <h4 className="sample-title font-display">{s.label}</h4>
                      <p className="sample-desc font-sans text-secondary">{s.description}</p>
                    </div>

                    <div className="sample-card-footer">
                      <button
                        className="btn btn-emerald btn-sm"
                        disabled={loading || analyzingSample !== null}
                        onClick={() => handleAnalyzeSample(s.name)}
                      >
                        {isAnalyzingThis ? (
                          <>
                            <RefreshCw size={13} className="animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            <Play size={13} />
                            ⚡ Run Pipeline &amp; Open
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Quick Engine Diagnostics */}
            <div className="engine-diagnostics-box font-mono">
              <div className="diag-item">
                <span className="diag-label">WELCH FFT:</span>
                <span className="diag-val text-cyan">4096-pt Blackman</span>
              </div>
              <div className="diag-item">
                <span className="diag-label">CUMULANT ML:</span>
                <span className="diag-val text-emerald">C40 / C42 Tensor</span>
              </div>
              <div className="diag-item">
                <span className="diag-label">COSTAS LOCK:</span>
                <span className="diag-val text-emerald">4th-Power Active</span>
              </div>
            </div>
          </div>
        </section>

        {/* Row 2: Recent Mission Telemetry Log */}
        <section className="missions-section glass-panel tactical-border">
          <div className="card-header-bar">
            <div className="card-title-group">
              <Layers size={16} className="text-cyan" />
              <h3 className="card-heading font-display">Mission Telemetry &amp; Ingestion History</h3>
            </div>
            <div className="mission-stats-summary font-mono">
              <span className="text-tertiary">TOTAL MISSIONS:</span>
              <span className="text-cyan font-bold">{missions.length}</span>
            </div>
          </div>

          <div className="table-wrapper">
            <table className="sigint-table font-mono">
              <thead>
                <tr>
                  <th>MISSION ID</th>
                  <th>SIGNAL / CAPTURE</th>
                  <th>SOURCE</th>
                  <th>TIMESTAMP</th>
                  <th>STATUS</th>
                  <th>MODULATION</th>
                  <th>EST. SNR</th>
                  <th>SAMPLES</th>
                  <th>BITS</th>
                  <th style={{ textAlign: 'right' }}>ACTION</th>
                </tr>
              </thead>
              <tbody>
                {missions.map((m) => (
                  <tr key={m.id} className="table-row">
                    <td className="text-cyan font-bold">{m.id.substring(0, 16)}</td>
                    <td className="font-display font-medium text-white">{m.filename}</td>
                    <td className="text-secondary">{m.source}</td>
                    <td className="text-tertiary">{m.timestamp}</td>
                    <td>
                      <span className={`badge ${m.status === 'complete' ? 'badge-emerald' : 'badge-amber'}`}>
                        {m.status}
                      </span>
                    </td>
                    <td>
                      <span className="badge badge-cyan">{m.modulation}</span>
                    </td>
                    <td className="text-emerald">+{m.snr.toFixed(1)} dB</td>
                    <td>{m.samples.toLocaleString()}</td>
                    <td>{m.bits.toLocaleString()}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={() => handleOpenWorkspaceWithMission(m)}
                      >
                        Inspect
                        <ChevronRight size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
};

export default Dashboard;
