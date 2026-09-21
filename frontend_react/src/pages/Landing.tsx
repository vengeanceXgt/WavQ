import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Radio, ArrowRight, Waves, Cpu, BarChart3, Layers,
  Shield, Activity, Zap, Play, Terminal, Database, Sliders, CheckCircle2
} from 'lucide-react';
import './Landing.css';

type ModType = 'BPSK' | 'QPSK' | '16QAM' | 'CHIRP';

export const Landing: React.FC = () => {
  const navigate = useNavigate();
  const [activeMod, setActiveMod] = useState<ModType>('QPSK');
  const [snr, setSnr] = useState<number>(24);
  const [carrierFreq, setCarrierFreq] = useState<number>(433.92);
  const [selectedStage, setSelectedStage] = useState<number>(2);

  // Canvas ref for live oscilloscope + constellation
  const scopeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Real-time canvas simulation for interactive scope
  useEffect(() => {
    const canvas = scopeCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;

    const render = () => {
      time += 0.04;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const w = canvas.width;
      const h = canvas.height;
      const halfW = w / 2;
      const noiseAmp = Math.max(0.02, 1.2 / Math.sqrt(Math.max(1, snr)));

      // ── LEFT PANE: Oscilloscope Waveform ──
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, halfW - 10, h);
      ctx.clip();

      // Grid
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.08)';
      ctx.lineWidth = 1;
      for (let x = 0; x < halfW; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(halfW, y);
        ctx.stroke();
      }

      // Center baseline
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(halfW - 10, h / 2);
      ctx.stroke();

      // Signal Trace
      ctx.strokeStyle = '#00f0ff';
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur = 8;
      ctx.lineWidth = 2;
      ctx.beginPath();

      const numPoints = 120;
      for (let i = 0; i < numPoints; i++) {
        const x = (i / (numPoints - 1)) * (halfW - 20) + 10;
        const t = time * 2 + i * 0.12;
        let val = 0;

        if (activeMod === 'BPSK') {
          const bit = Math.floor((i + Math.floor(time * 3)) / 15) % 2 === 0 ? 1 : -1;
          val = bit * Math.cos(t * 1.5);
        } else if (activeMod === 'QPSK') {
          const sym = Math.floor((i + Math.floor(time * 4)) / 15) % 4;
          const phase = (sym * Math.PI) / 2 + Math.PI / 4;
          val = Math.cos(t * 1.5 + phase);
        } else if (activeMod === '16QAM') {
          const level = ((Math.floor((i + Math.floor(time * 4)) / 12) % 4) - 1.5) / 1.5;
          val = level * Math.cos(t * 1.8);
        } else {
          // Chirp
          const chirpFreq = 0.5 + (i / numPoints) * 3;
          val = Math.sin(t * chirpFreq);
        }

        const noise = (Math.random() - 0.5) * noiseAmp * 2;
        const y = h / 2 - (val + noise) * (h * 0.35);

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();

      // ── RIGHT PANE: Constellation Diagram ──
      ctx.save();
      const cx = halfW + (halfW / 2);
      const cy = h / 2;
      const radius = Math.min(halfW, h) * 0.4;

      // Reticle rings
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.12)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2);
      ctx.stroke();

      // Axis crosshairs
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.2)';
      ctx.beginPath();
      ctx.moveTo(cx - radius - 15, cy);
      ctx.lineTo(cx + radius + 15, cy);
      ctx.moveTo(cx, cy - radius - 15);
      ctx.lineTo(cx, cy + radius + 15);
      ctx.stroke();

      // Constellation points
      let idealPoints: [number, number][] = [];
      if (activeMod === 'BPSK') {
        idealPoints = [[-1, 0], [1, 0]];
      } else if (activeMod === 'QPSK') {
        const s = 0.707;
        idealPoints = [[s, s], [-s, s], [-s, -s], [s, -s]];
      } else if (activeMod === '16QAM') {
        const levels = [-0.95, -0.32, 0.32, 0.95];
        levels.forEach(i => {
          levels.forEach(q => idealPoints.push([i, q]));
        });
      } else {
        // Chirp: circular orbit
        for (let a = 0; a < 8; a++) {
          const ang = (a / 8) * Math.PI * 2 + time;
          idealPoints.push([Math.cos(ang) * 0.8, Math.sin(ang) * 0.8]);
        }
      }

      // Draw cloud of samples
      const samplesPerSymbol = activeMod === '16QAM' ? 2 : 8;
      idealPoints.forEach(([ix, iq]) => {
        // Draw ideal symbol marker
        ctx.fillStyle = '#00ff9d';
        ctx.shadowColor = '#00ff9d';
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(cx + ix * radius, cy - iq * radius, 3.5, 0, Math.PI * 2);
        ctx.fill();

        // Draw noise jitter cloud
        ctx.fillStyle = 'rgba(0, 240, 255, 0.55)';
        ctx.shadowBlur = 0;
        for (let s = 0; s < samplesPerSymbol; s++) {
          const jx = ix + (Math.random() - 0.5) * noiseAmp * 0.8;
          const jq = iq + (Math.random() - 0.5) * noiseAmp * 0.8;
          ctx.beginPath();
          ctx.arc(cx + jx * radius, cy - jq * radius, 1.8, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      ctx.restore();

      animFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [activeMod, snr]);

  const PIPELINE_DATA = [
    {
      num: 1,
      name: 'Ingestion & DC Removal',
      icon: Database,
      tag: 'I/Q CALIBRATION',
      desc: 'Parses raw interleaved float32/int16 byte streams. Calculates quadrature balance, dynamic headroom, and suppresses DC carrier feedthrough.',
      stats: 'Throughput: >150 MSps | Precision: IEEE-754'
    },
    {
      num: 2,
      name: 'Spectral FFT & CFO',
      icon: Activity,
      tag: 'WELCH ESTIMATOR',
      desc: 'Calculates high-resolution Power Spectral Density (PSD). Detects 3dB/10dB occupied bandwidth and extracts non-linear Carrier Frequency Offset.',
      stats: 'FFT Window: 4096 Blackman-Harris | CFO Accuracy: ±12 Hz'
    },
    {
      num: 3,
      name: 'AMC Classifier',
      icon: Cpu,
      tag: 'ML CUMULANT TENSOR',
      desc: 'Computes higher-order statistics (C40, C42, kurtosis). Ranks candidate modulations (BPSK, QPSK, 8PSK, 16QAM, FSK) with decision boundaries.',
      stats: 'Confidence: 99.4% | Latency: 4.8 ms'
    },
    {
      num: 4,
      name: 'Carrier & Clock Sync',
      icon: Waves,
      tag: 'COSTAS & GARDNER',
      desc: 'Closes Gardner timing recovery loop to track symbol clock jitter. Locks carrier phase using 4th-power Costas loop to eliminate constellation spinning.',
      stats: 'Jitter: < 0.015 Ts | Damping: ζ = 0.707'
    },
    {
      num: 5,
      name: 'Symbol Demodulation',
      icon: BarChart3,
      tag: 'MAX-LOG-MAP LLR',
      desc: 'Projects constellation points into soft Log-Likelihood Ratio (LLR) bit planes and performs hard symbol decision slicing into raw binary stream.',
      stats: 'Bit Stream: 100% Sliced | Soft Margins Active'
    },
    {
      num: 6,
      name: 'Frame & Protocol Sync',
      icon: Layers,
      tag: 'SYNCWORD CORRELATOR',
      desc: 'Performs sliding auto-correlation across bitstream to detect repeating frame lengths, preamble synchronization words, and header boundaries.',
      stats: 'Autocorrelation Peak: 0.982 | Framing: Locked'
    },
    {
      num: 7,
      name: 'FEC & Intelligence Dossier',
      icon: Shield,
      tag: 'AIR-GAPPED TELEMETRY',
      desc: 'Executes syndrome parity checks, estimates Bit Error Rate (BER), and generates cryptographically signed JSON & symbol mission archives.',
      stats: 'Syndrome Check: Zero Errors | SHA-256 Provenance'
    },
  ];

  return (
    <div className="landing-root">
      {/* Background Ambience */}
      <div className="landing-ambient-glow" />
      <div className="landing-grid-overlay" />

      {/* Top Telemetry Header */}
      <header className="landing-top-bar">
        <div className="brand-zone">
          <div className="brand-glyph">
            <Radio size={20} className="text-cyan animate-pulse" />
          </div>
          <div className="brand-text">
            <span className="brand-title font-display">NTROv5</span>
            <span className="brand-sub font-mono">SIGINT SPEC-OPS</span>
          </div>
        </div>

        <div className="telemetry-bar font-mono">
          <div className="telemetry-item">
            <span className="telemetry-label">DSP CLUSTER:</span>
            <span className="telemetry-val text-emerald"><span className="pulse-dot" /> ACTIVE</span>
          </div>
          <div className="telemetry-divider">/</div>
          <div className="telemetry-item">
            <span className="telemetry-label">TUNER BAND:</span>
            <span className="telemetry-val text-cyan">20 MHz – 6.0 GHz</span>
          </div>
          <div className="telemetry-divider">/</div>
          <div className="telemetry-item">
            <span className="telemetry-label">SECURITY:</span>
            <span className="telemetry-val text-emerald">AIR-GAPPED (LOCAL)</span>
          </div>
        </div>

        <div className="header-actions">
          <button className="btn btn-secondary btn-sm" onClick={() => navigate('/dashboard')}>
            Dashboard
          </button>
          <button className="btn btn-primary btn-sm" onClick={() => navigate('/workspace')}>
            Launch Workstation
            <ArrowRight size={14} />
          </button>
        </div>
      </header>

      {/* Main Hero Container */}
      <main className="landing-main-container">
        <section className="hero-section">
          <div className="hero-badge-pill">
            <span className="badge badge-cyan">
              <Activity size={12} /> PRECISION RF TELEMETRY INSTRUMENT
            </span>
            <span className="badge badge-emerald">
              VERSION 5.4 STABLE
            </span>
          </div>

          <h1 className="hero-main-title">
            Signal Intelligence &amp; <br />
            <span className="gradient-text">Modulation Demodulation</span> Workstation
          </h1>

          <p className="hero-lead-text">
            Engineered specifically for <strong>.IQ</strong> and <strong>.WAV</strong> telemetry analysis.
            Featuring automated multi-cumulant modulation classification, high-speed Welch spectral decomposition,
            Costas carrier synchronization, and protocol frame reconstruction in a defense-grade instrument.
          </p>

          <div className="hero-cta-deck">
            <button
              className="btn btn-primary btn-lg"
              onClick={() => navigate('/dashboard')}
            >
              <Zap size={18} />
              Open Telemetry Dashboard
              <ArrowRight size={18} />
            </button>

            <button
              className="btn btn-secondary btn-lg"
              onClick={() => navigate('/workspace')}
            >
              <Terminal size={18} />
              Direct Workstation Mode
            </button>

            <button
              className="btn btn-ghost btn-lg sample-trigger-btn font-mono"
              onClick={() => navigate('/dashboard')}
              title="Load instant pre-recorded test signal"
            >
              <Play size={16} className="text-emerald" />
              ⚡ Instant 1-Click Sample Demo
            </button>
          </div>

          <div className="file-badges-row">
            <span className="support-label font-mono">SUPPORTED INGESTION FORMATS:</span>
            <span className="badge badge-cyan">.IQ COMPLEX FLOAT32</span>
            <span className="badge badge-emerald">.WAV AUDIO BASEBAND</span>
            <span className="badge badge-gray">.DAT RAW BINARY</span>
            <span className="badge badge-gray">.SIGMF COMPLIANT</span>
          </div>
        </section>

        {/* ── INTERACTIVE LIVE SIGNAL SCOPE LAB ── */}
        <section className="live-lab-card glass-panel tactical-border">
          <div className="lab-header">
            <div className="lab-title-group">
              <div className="lab-icon-badge">
                <Activity size={18} className="text-cyan" />
              </div>
              <div>
                <h3 className="lab-title font-display">Interactive Signal Synthesizer &amp; Scope</h3>
                <p className="lab-subtitle font-mono">Real-time simulation of carrier modulation, I/Q constellation, and SNR degradation</p>
              </div>
            </div>

            <div className="mod-selector-tabs">
              {(['BPSK', 'QPSK', '16QAM', 'CHIRP'] as ModType[]).map((mod) => (
                <button
                  key={mod}
                  className={`mod-tab-btn font-mono ${activeMod === mod ? 'active' : ''}`}
                  onClick={() => setActiveMod(mod)}
                >
                  {mod}
                </button>
              ))}
            </div>
          </div>

          <div className="lab-canvas-wrapper">
            <div className="canvas-labels font-mono">
              <span className="scope-pane-tag">TIME-DOMAIN OSCILLOSCOPE (I/Q)</span>
              <span className="scope-pane-tag">CONSTELLATION POLAR DIAGRAM</span>
            </div>
            <canvas
              ref={scopeCanvasRef}
              width={820}
              height={260}
              className="scope-canvas"
            />
          </div>

          <div className="lab-controls-bar">
            <div className="lab-slider-group">
              <div className="slider-header font-mono">
                <span className="slider-name"><Sliders size={13} /> Carrier Center Freq:</span>
                <span className="slider-val text-cyan">{carrierFreq.toFixed(2)} MHz</span>
              </div>
              <input
                type="range"
                min="100"
                max="900"
                step="0.5"
                value={carrierFreq}
                onChange={(e) => setCarrierFreq(parseFloat(e.target.value))}
                className="tactical-slider"
              />
            </div>

            <div className="lab-slider-group">
              <div className="slider-header font-mono">
                <span className="slider-name"><Activity size={13} /> Signal-to-Noise Ratio (SNR):</span>
                <span className={`slider-val ${snr > 15 ? 'text-emerald' : 'text-amber'}`}>{snr} dB</span>
              </div>
              <input
                type="range"
                min="2"
                max="40"
                value={snr}
                onChange={(e) => setSnr(parseInt(e.target.value, 10))}
                className="tactical-slider"
              />
            </div>

            <div className="lab-action-area">
              <button
                className="btn btn-emerald btn-sm"
                onClick={() => navigate('/workspace')}
              >
                Analyze in Workstation
                <ArrowRight size={14} />
              </button>
            </div>
          </div>
        </section>

        {/* ── PIPELINE ARCHITECTURE ROADMAP ── */}
        <section className="pipeline-section">
          <div className="section-head">
            <span className="badge badge-cyan font-mono">ORCHESTRATION PIPELINE</span>
            <h2 className="section-title">7-Stage Real-Time DSP Signal Pipeline</h2>
            <p className="section-desc">
              Every signal passes through an uncompromised chain of rigorous algorithmic analysis stages from raw byte characterization to protocol extraction.
            </p>
          </div>

          <div className="pipeline-steps-container">
            <div className="pipeline-stepper-nav">
              {PIPELINE_DATA.map((stage) => {
                const Icon = stage.icon;
                const isSelected = selectedStage === stage.num;
                return (
                  <button
                    key={stage.num}
                    className={`pipeline-nav-btn ${isSelected ? 'selected' : ''}`}
                    onClick={() => setSelectedStage(stage.num)}
                  >
                    <div className="stage-num-badge font-mono">
                      <Icon size={14} />
                    </div>
                    <div className="stage-btn-meta">
                      <span className="stage-btn-title font-display">{stage.name}</span>
                      <span className="stage-btn-tag font-mono">{stage.tag}</span>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="pipeline-detail-card glass-panel tactical-border">
              {(() => {
                const stage = PIPELINE_DATA.find((s) => s.num === selectedStage) || PIPELINE_DATA[0];
                const Icon = stage.icon;
                return (
                  <div className="stage-detail-content">
                    <div className="stage-detail-header">
                      <div className="stage-hero-icon">
                        <Icon size={28} className="text-cyan" />
                      </div>
                      <div>
                        <div className="stage-micro-tag font-mono text-emerald">STAGE 0{stage.num} // {stage.tag}</div>
                        <h3 className="stage-detail-title font-display">{stage.name}</h3>
                      </div>
                    </div>

                    <p className="stage-body-text">{stage.desc}</p>

                    <div className="stage-telemetry-box font-mono">
                      <CheckCircle2 size={16} className="text-emerald" />
                      <span>{stage.stats}</span>
                    </div>

                    <div className="stage-actions">
                      <button className="btn btn-secondary btn-sm" onClick={() => navigate('/dashboard')}>
                        Test in Ingestion
                      </button>
                      <button className="btn btn-primary btn-sm" onClick={() => navigate('/workspace')}>
                        Open in Workstation
                        <ArrowRight size={14} />
                      </button>
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        </section>

        {/* ── DEFENSE & PRIVACY ARCHITECTURE ── */}
        <section className="security-section glass-panel">
          <div className="security-grid">
            <div className="security-col">
              <div className="sec-icon-circle">
                <Shield size={22} className="text-emerald" />
              </div>
              <h4 className="font-display">Air-Gapped &amp; Zero External Telemetry</h4>
              <p className="font-sans">
                Signal analysis runs 100% on localhost within isolated in-memory circular buffers.
                Zero telemetry leaves your machine, ensuring full compliance with sensitive defense and intelligence protocols.
              </p>
            </div>

            <div className="security-col">
              <div className="sec-icon-circle">
                <Cpu size={22} className="text-cyan" />
              </div>
              <h4 className="font-display">Hardware-Accelerated DSP Numerics</h4>
              <p className="font-sans">
                Vectorized NumPy and SciPy signal kernels maximize SIMD utilization on AVX2/NEON architectures,
                demodulating up to 150 MSps with microsecond-level synchronization lock.
              </p>
            </div>

            <div className="security-col">
              <div className="sec-icon-circle">
                <Database size={22} className="text-amber" />
              </div>
              <h4 className="font-display">Cryptographic Provenance</h4>
              <p className="font-sans">
                Every processed signal is issued an immutable SHA-256 hash log and serialized JSON schema,
                guaranteeing repeatable forensic verification across mission archives.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-left font-mono">
          <span>NTROv5 SIGNAL INTELLIGENCE SUITE</span>
          <span className="footer-pipe">|</span>
          <span className="text-secondary">SYSTEM ID: SIGINT-STN-09</span>
        </div>
        <div className="footer-right">
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/dashboard')}>
            Dashboard
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/workspace')}>
            Workstation
          </button>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
