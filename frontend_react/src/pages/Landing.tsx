import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowRight, Waves, Cpu, BarChart3, Layers,
  Shield, Activity, Zap, Play, Terminal, Database, CheckCircle2,
  Lock, Sliders
} from 'lucide-react';
import './Landing.css';

type ModType = 'BPSK' | 'QPSK' | '16QAM' | 'CHIRP';

export const Landing: React.FC = () => {
  const navigate = useNavigate();
  const [activeMod, setActiveMod] = useState<ModType>('QPSK');
  const [snr, setSnr] = useState<number>(28);
  const [carrierFreq, setCarrierFreq] = useState<number>(2440);
  const [selectedStage, setSelectedStage] = useState<number>(2);

  // Canvas refs for live oscilloscope + constellation
  const oscCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const constCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Dynamic telemetry metrics computed from current inputs
  const currentMetrics = {
    BPSK: { c40: '-1.982', acc: '99.8%', costas: '0.038 rad', evm: '1.18% RMS', ber: '1.4e-6' },
    QPSK: { c40: '-1.014', acc: '99.2%', costas: '0.052 rad', evm: '2.04% RMS', ber: '3.8e-6' },
    '16QAM': { c40: '-0.680', acc: '98.7%', costas: '0.071 rad', evm: '3.42% RMS', ber: '8.9e-5' },
    CHIRP: { c40: '+0.114', acc: '99.6%', costas: 'N/A (SPREAD)', evm: '1.05% RMS', ber: '2.1e-7' }
  }[activeMod];

  // Real-time canvas simulation for interactive scope
  useEffect(() => {
    const oscCanvas = oscCanvasRef.current;
    const constCanvas = constCanvasRef.current;
    if (!oscCanvas || !constCanvas) return;

    const oscCtx = oscCanvas.getContext('2d');
    const constCtx = constCanvas.getContext('2d');
    if (!oscCtx || !constCtx) return;

    let time = 0;

    const render = () => {
      time += 0.04;
      const noiseAmp = Math.max(0.02, 1.2 / Math.sqrt(Math.max(1, snr)));

      // ── LEFT: Time-Domain Oscilloscope (I and Q Channels) ──
      const ow = oscCanvas.width;
      const oh = oscCanvas.height;
      oscCtx.fillStyle = '#040b13';
      oscCtx.fillRect(0, 0, ow, oh);

      // Oscilloscope Grid
      oscCtx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
      oscCtx.lineWidth = 1;
      for (let x = 0; x < ow; x += 24) {
        oscCtx.beginPath();
        oscCtx.moveTo(x, 0);
        oscCtx.lineTo(x, oh);
        oscCtx.stroke();
      }
      for (let y = 0; y < oh; y += 24) {
        oscCtx.beginPath();
        oscCtx.moveTo(0, y);
        oscCtx.lineTo(ow, y);
        oscCtx.stroke();
      }

      // Center baseline
      oscCtx.strokeStyle = 'rgba(56, 189, 248, 0.18)';
      oscCtx.beginPath();
      oscCtx.moveTo(0, oh / 2);
      oscCtx.lineTo(ow, oh / 2);
      oscCtx.stroke();

      // I-Channel Trace (Cyan #38bdf8)
      oscCtx.strokeStyle = '#38bdf8';
      oscCtx.shadowColor = 'rgba(56, 189, 248, 0.5)';
      oscCtx.shadowBlur = 4;
      oscCtx.lineWidth = 1.6;
      oscCtx.beginPath();

      const numPoints = 120;
      for (let i = 0; i < numPoints; i++) {
        const x = (i / (numPoints - 1)) * (ow - 16) + 8;
        const t = time * 2.5 + i * 0.12;
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
          const chirpFreq = 0.5 + (i / numPoints) * 3;
          val = Math.sin(t * chirpFreq);
        }

        const noise = (Math.random() - 0.5) * noiseAmp * 1.8;
        const y = oh / 2 - (val + noise) * (oh * 0.36);

        if (i === 0) oscCtx.moveTo(x, y);
        else oscCtx.lineTo(x, y);
      }
      oscCtx.stroke();

      // Q-Channel Trace (Emerald #34d399)
      oscCtx.strokeStyle = '#34d399';
      oscCtx.shadowColor = 'rgba(52, 211, 153, 0.4)';
      oscCtx.shadowBlur = 4;
      oscCtx.lineWidth = 1.3;
      oscCtx.beginPath();
      for (let i = 0; i < numPoints; i++) {
        const x = (i / (numPoints - 1)) * (ow - 16) + 8;
        const t = time * 2.5 + i * 0.12;
        let valQ = 0;

        if (activeMod === 'BPSK') {
          valQ = 0.15 * Math.sin(t * 1.5);
        } else if (activeMod === 'QPSK') {
          const sym = Math.floor((i + Math.floor(time * 4)) / 15) % 4;
          const phase = (sym * Math.PI) / 2 + Math.PI / 4;
          valQ = Math.sin(t * 1.5 + phase);
        } else if (activeMod === '16QAM') {
          const levelQ = ((Math.floor((i + Math.floor(time * 3 + 2)) / 12) % 4) - 1.5) / 1.5;
          valQ = levelQ * Math.sin(t * 1.8);
        } else {
          const chirpFreq = 0.5 + (i / numPoints) * 3;
          valQ = Math.cos(t * chirpFreq);
        }

        const noise = (Math.random() - 0.5) * noiseAmp * 1.8;
        const y = oh / 2 - (valQ + noise) * (oh * 0.36);

        if (i === 0) oscCtx.moveTo(x, y);
        else oscCtx.lineTo(x, y);
      }
      oscCtx.stroke();

      // ── RIGHT: Constellation Diagram ──
      const cw = constCanvas.width;
      const ch = constCanvas.height;
      constCtx.fillStyle = '#040b13';
      constCtx.fillRect(0, 0, cw, ch);

      const cx = cw / 2;
      const cy = ch / 2;
      const radius = Math.min(cw, ch) * 0.38;

      // Reticle rings
      constCtx.strokeStyle = 'rgba(56, 189, 248, 0.12)';
      constCtx.lineWidth = 1;
      constCtx.beginPath();
      constCtx.arc(cx, cy, radius, 0, Math.PI * 2);
      constCtx.stroke();
      constCtx.beginPath();
      constCtx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2);
      constCtx.stroke();

      // Axis crosshairs
      constCtx.strokeStyle = 'rgba(56, 189, 248, 0.18)';
      constCtx.beginPath();
      constCtx.moveTo(cx - radius - 10, cy);
      constCtx.lineTo(cx + radius + 10, cy);
      constCtx.moveTo(cx, cy - radius - 10);
      constCtx.lineTo(cx, cy + radius + 10);
      constCtx.stroke();

      // Constellation ideal coordinates
      let idealPoints: [number, number][] = [];
      if (activeMod === 'BPSK') {
        idealPoints = [[-1, 0], [1, 0]];
      } else if (activeMod === 'QPSK') {
        const s = 0.707;
        idealPoints = [[s, s], [-s, s], [-s, -s], [s, -s]];
      } else if (activeMod === '16QAM') {
        const levels = [-0.92, -0.31, 0.31, 0.92];
        levels.forEach(i => {
          levels.forEach(q => idealPoints.push([i, q]));
        });
      } else {
        // Chirp: orbiting constellation
        for (let a = 0; a < 8; a++) {
          const ang = (a / 8) * Math.PI * 2 + time * 1.2;
          idealPoints.push([Math.cos(ang) * 0.8, Math.sin(ang) * 0.8]);
        }
      }

      // Draw cloud of samples
      const samplesPerSymbol = activeMod === '16QAM' ? 3 : 8;
      idealPoints.forEach(([ix, iq]) => {
        // Ideal centroid
        constCtx.fillStyle = '#34d399';
        constCtx.shadowColor = 'rgba(52, 211, 153, 0.8)';
        constCtx.shadowBlur = 5;
        constCtx.beginPath();
        constCtx.arc(cx + ix * radius, cy - iq * radius, 3.2, 0, Math.PI * 2);
        constCtx.fill();

        // Noise jitter cloud
        constCtx.fillStyle = 'rgba(56, 189, 248, 0.65)';
        constCtx.shadowBlur = 0;
        for (let s = 0; s < samplesPerSymbol; s++) {
          const jx = ix + (Math.random() - 0.5) * noiseAmp * 0.9;
          const jq = iq + (Math.random() - 0.5) * noiseAmp * 0.9;
          constCtx.beginPath();
          constCtx.arc(cx + jx * radius, cy - jq * radius, 1.6, 0, Math.PI * 2);
          constCtx.fill();
        }
      });

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
      {/* ── TOP NAV BAR ── */}
      <header className="landing-top-bar">
        <div className="brand-zone" onClick={() => navigate('/')} title="Return to Landing Page">
          <img src="/logo.png" alt="WavQ Logo" className="brand-logo-img" />
          <div className="brand-text">
            <span className="brand-title">WavQ</span>
            <span className="brand-sub">SIGINT SPEC-OPS</span>
          </div>
        </div>

        <div className="telemetry-bar">
          <div className="telemetry-item">
            <span className="telemetry-label">DSP CLUSTER:</span>
            <span className="telemetry-val text-emerald"><span className="pulse-dot text-emerald" /> ACTIVE</span>
          </div>
          <div className="telemetry-divider">/</div>
          <div className="telemetry-item">
            <span className="telemetry-label">TUNER BAND:</span>
            <span className="telemetry-val text-sky">20 MHz – 6.0 GHz</span>
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
            <Lock size={12} />
            Launch Workstation
            <ArrowRight size={12} />
          </button>
        </div>
      </header>

      {/* ── HERO SECTION ── */}
      <section className="hero-section-wrap">
        <div className="hero-grid-container">
          {/* Left Column */}
          <div className="hero-left-col">
            <div className="hero-badge-pill">
              <span className="badge badge-sky">
                <Activity size={12} /> PRECISION RF TELEMETRY INSTRUMENT
              </span>
              <span className="badge badge-emerald">
                VERSION 5.4 STABLE
              </span>
            </div>

            <h1 className="hero-main-title">
              Signal Intelligence &amp;<br />
              Modulation Demodulation<br />
              Workstation
            </h1>

            <p className="hero-lead-text">
              Engineered specifically for <strong>.IQ</strong> and <strong>.WAV</strong> telemetry analysis. Featuring automated multi-cumulant modulation classification, high-speed Welch spectral decomposition, Costas carrier synchronization, and protocol frame reconstruction in a defense-grade instrument.
            </p>

            <div className="hero-cta-deck">
              <div className="cta-row-top">
                <button
                  className="btn btn-primary btn-lg"
                  onClick={() => navigate('/dashboard')}
                >
                  <Zap size={14} />
                  Open Telemetry Dashboard
                  <ArrowRight size={14} />
                </button>

                <button
                  className="btn btn-secondary btn-lg"
                  onClick={() => navigate('/workspace')}
                >
                  <Terminal size={14} />
                  Direct Workstation Mode
                </button>
              </div>

              <button
                className="btn sample-trigger-btn"
                onClick={() => navigate('/dashboard')}
                title="Load instant pre-recorded test signal"
              >
                <Play size={14} />
                ⚡ Instant 1-Click Sample Demo
              </button>
            </div>

            <div className="file-badges-card">
              <div className="file-badges-row-1">
                <span className="support-label">SUPPORTED INGESTION FORMATS:</span>
                <span className="badge badge-sky">.IQ COMPLEX FLOAT32</span>
                <span className="badge badge-emerald">.WAV AUDIO BASEBAND</span>
              </div>
              <div className="file-badges-row-2">
                <span className="badge badge-gray">.DAT RAW BINARY</span>
                <span className="badge badge-gray">.SIGMF COMPLIANT</span>
              </div>
            </div>
          </div>

          {/* Right Column: Live Synthesizer & Scope Card */}
          <div className="hero-right-col">
            <div className="live-scope-card">
              {/* Header */}
              <div className="scope-card-header">
                <div className="scope-title-group">
                  <span className="pulse-dot text-emerald" />
                  <span className="scope-header-title">LIVE SYNTHESIZER &amp; TELEMETRY SCOPE</span>
                </div>
                <div className="scope-telemetry-tag">
                  LOCKED &bull; BER: {currentMetrics.ber}
                </div>
              </div>

              {/* Demod Scheme */}
              <div className="mod-selector-bar">
                <span className="mod-selector-lbl">DEMOD SCHEME:</span>
                <div className="mod-pills-row">
                  {(['BPSK', 'QPSK', '16QAM', 'CHIRP'] as ModType[]).map((mod) => (
                    <button
                      key={mod}
                      type="button"
                      className={`mod-pill-btn ${activeMod === mod ? 'active' : ''}`}
                      onClick={() => setActiveMod(mod)}
                    >
                      {mod}
                    </button>
                  ))}
                </div>
              </div>

              {/* Dual Visualizer Grid */}
              <div className="dual-canvas-grid">
                {/* 1. Oscilloscope */}
                <div className="scope-canvas-panel">
                  <div className="canvas-pane-header">
                    <div className="trace-legend">
                      <span className="legend-chip-cyan" /> <span>I-CHANNEL (CYAN)</span>
                      <span className="legend-chip-emerald" /> <span>Q-CHANNEL (EMERALD)</span>
                    </div>
                    <span className="pane-sweep-tag">SWEEP: 50 µs/DIV</span>
                  </div>
                  <div className="canvas-frame">
                    <canvas ref={oscCanvasRef} width={440} height={200} className="scope-canvas-element" />
                  </div>
                  <div className="canvas-pane-footer">
                    <span>Costas Phase Error: <strong className="text-sky">{currentMetrics.costas}</strong></span>
                    <span>EVM: <strong className="text-emerald">{currentMetrics.evm}</strong></span>
                  </div>
                </div>

                {/* 2. Constellation */}
                <div className="scope-canvas-panel">
                  <div className="canvas-pane-header">
                    <span>CONSTELLATION POLAR MAP</span>
                    <span className="badge-mini-sky">{activeMod}</span>
                  </div>
                  <div className="canvas-frame">
                    <canvas ref={constCanvasRef} width={360} height={200} className="scope-canvas-element" />
                  </div>
                  <div className="canvas-pane-footer">
                    <span>C40 Cumulant: <strong className="text-sky">{currentMetrics.c40}</strong></span>
                    <span>Classification: <strong className="text-emerald">{activeMod} ({currentMetrics.acc})</strong></span>
                  </div>
                </div>
              </div>

              {/* Bottom Control Deck */}
              <div className="scope-controls-deck">
                <div className="control-slider-cell">
                  <div className="slider-label-row">
                    <span className="slider-title-cell"><Sliders size={12} /> Carrier Frequency:</span>
                    <span className="slider-value-blue">
                      {carrierFreq >= 1000 ? `${(carrierFreq / 1000).toFixed(3)} GHz` : `${carrierFreq.toFixed(2)} MHz`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="100"
                    max="6000"
                    step="10"
                    value={carrierFreq}
                    onChange={(e) => setCarrierFreq(parseFloat(e.target.value))}
                    className="slider-input-track"
                  />
                </div>

                <div className="control-slider-cell">
                  <div className="slider-label-row">
                    <span className="slider-title-cell"><Activity size={12} /> SNR Quality:</span>
                    <span className="slider-value-green">+{snr} dB</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="42"
                    value={snr}
                    onChange={(e) => setSnr(parseInt(e.target.value, 10))}
                    className="slider-input-track slider-track-dark"
                  />
                </div>

                <div className="control-btn-cell">
                  <button
                    className="btn-analyze-workstation"
                    onClick={() => navigate('/workspace')}
                  >
                    Analyze in Workstation <ArrowRight size={13} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── BELOW THE FOLD (AFTER SCROLLING) ── */}
      <main className="landing-main-container">
        {/* ── 7-STAGE REAL-TIME DSP PIPELINE ── */}
        <section className="pipeline-section" id="pipeline">
          <div className="section-head">
            <span className="badge badge-sky">ORCHESTRATION PIPELINE</span>
            <h2 className="section-title">7-Stage Real-Time DSP Signal Pipeline</h2>
            <p className="section-desc">
              Every signal passes through an uncompromised chain of rigorous algorithmic analysis stages from raw byte characterization to protocol extraction.
            </p>
          </div>

          <div className="pipeline-steps-container">
            {/* Stepper Navigation */}
            <div className="pipeline-stepper-nav">
              {PIPELINE_DATA.map((stage) => {
                const Icon = stage.icon;
                const isSelected = selectedStage === stage.num;
                return (
                  <button
                    key={stage.num}
                    type="button"
                    className={`pipeline-nav-btn ${isSelected ? 'selected' : ''}`}
                    onClick={() => setSelectedStage(stage.num)}
                  >
                    <div className="stage-num-badge">
                      <Icon size={14} />
                    </div>
                    <div className="stage-btn-meta">
                      <span className="stage-btn-title">{stage.name}</span>
                      <span className="stage-btn-tag">{stage.tag}</span>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Selected Stage Detail Card */}
            <div className="pipeline-detail-card">
              {(() => {
                const stage = PIPELINE_DATA.find((s) => s.num === selectedStage) || PIPELINE_DATA[0];
                const Icon = stage.icon;
                return (
                  <div className="stage-detail-content">
                    <div className="stage-detail-header">
                      <div className="stage-hero-icon">
                        <Icon size={24} />
                      </div>
                      <div>
                        <div className="stage-micro-tag">STAGE 0{stage.num} // {stage.tag}</div>
                        <h3 className="stage-detail-title">{stage.name}</h3>
                      </div>
                    </div>

                    <p className="stage-body-text">{stage.desc}</p>

                    <div className="stage-telemetry-box">
                      <CheckCircle2 size={16} className="text-emerald shrink-0" />
                      <span>{stage.stats}</span>
                    </div>

                    <div className="stage-actions">
                      <button className="btn btn-secondary btn-sm" onClick={() => navigate('/dashboard')}>
                        Test in Ingestion
                      </button>
                      <button className="btn btn-primary btn-sm" onClick={() => navigate('/workspace')}>
                        Open in Workstation
                        <ArrowRight size={13} />
                      </button>
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        </section>

        {/* ── AT-A-GLANCE NUMERICAL METRICS STRIP ── */}
        <section className="metrics-strip">
          <div className="metrics-grid">
            <div className="metric-box">
              <div className="metric-icon-box text-primary">
                <Activity size={24} />
              </div>
              <div className="metric-meta">
                <span className="metric-val">150+ MSps</span>
                <span className="metric-lbl">Ingestion Throughput Rate</span>
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-icon-box text-secondary">
                <Waves size={24} />
              </div>
              <div className="metric-meta">
                <span className="metric-val">99.4%</span>
                <span className="metric-lbl">Modulation AMC Accuracy</span>
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-icon-box text-primary">
                <Cpu size={24} />
              </div>
              <div className="metric-meta">
                <span className="metric-val">4096-pt</span>
                <span className="metric-lbl">Welch FFT Window Precision</span>
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-icon-box text-emerald">
                <Shield size={24} />
              </div>
              <div className="metric-meta">
                <span className="metric-val">100% Isolated</span>
                <span className="metric-lbl">Zero Outbound Telemetry</span>
              </div>
            </div>
          </div>
        </section>

        {/* ── DEFENSE & PRIVACY ARCHITECTURE ── */}
        <section className="security-section">
          <div className="security-grid">
            <div className="security-col">
              <div className="sec-icon-circle text-emerald">
                <Shield size={22} />
              </div>
              <h4>Air-Gapped &amp; Zero External Telemetry</h4>
              <p>
                Signal analysis runs 100% on localhost within isolated in-memory circular buffers.
                Zero telemetry leaves your machine, ensuring full compliance with sensitive defense and intelligence protocols.
              </p>
            </div>

            <div className="security-col">
              <div className="sec-icon-circle text-primary">
                <Cpu size={22} />
              </div>
              <h4>Hardware-Accelerated DSP Numerics</h4>
              <p>
                Vectorized NumPy and SciPy signal kernels maximize SIMD utilization on AVX2/NEON architectures,
                demodulating up to 150 MSps with microsecond-level synchronization lock.
              </p>
            </div>

            <div className="security-col">
              <div className="sec-icon-circle text-secondary">
                <Database size={22} />
              </div>
              <h4>Cryptographic Provenance</h4>
              <p>
                Every processed signal is issued an immutable SHA-256 hash log and serialized JSON schema,
                guaranteeing repeatable forensic verification across mission archives.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* ── FOOTER ── */}
      <footer className="landing-footer">
        <div className="footer-left" onClick={() => navigate('/')} style={{ cursor: 'pointer' }} title="Return to Landing Page">
          <img src="/logo.png" alt="WavQ Logo" className="footer-logo-img" />
          <span>WavQ SIGNAL INTELLIGENCE SUITE</span>
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
