// main.js – Frontend logic for NTROv5 workstation
// Handles drag‑and‑drop, file upload, processing animation, and panel creation

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const workspace = document.getElementById('workspace');
const statusEl = document.getElementById('status');

// ---------- Utility ----------
function setStatus(text, color = 'var(--color-text-secondary)') {
  statusEl.textContent = text;
  statusEl.style.color = color;
}

function animateDropZone(state) {
  // state: 'idle' | 'hover' | 'dragover'
  if (state === 'hover') {
    dropZone.classList.add('hover');
    dropZone.classList.remove('dragover');
  } else if (state === 'dragover') {
    dropZone.classList.add('dragover');
    dropZone.classList.remove('hover');
  } else {
    dropZone.classList.remove('hover', 'dragover');
  }
}

// ---------- Drag & Drop ----------
['dragenter', 'dragover'].forEach(ev => {
  dropZone.addEventListener(ev, e => {
    e.preventDefault();
    animateDropZone('dragover');
  });
});
['dragleave', 'drop'].forEach(ev => {
  dropZone.addEventListener(ev, e => {
    e.preventDefault();
    animateDropZone('idle');
  });
});

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});

dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// ---------- Processing animation sequence ----------
const stages = [
  {label: 'FILE DETECTED', duration: 600},
  {label: 'READING SIGNAL', duration: 800},
  {label: 'VALIDATING DATA', duration: 600},
  {label: 'BUILDING WAVEFORM', duration: 800},
  {label: 'COMPUTING SPECTRUM', duration: 800},
  {label: 'MAPPING FREQUENCY', duration: 600},
  {label: 'EXTRACTING FEATURES', duration: 800},
  {label: 'ANALYSIS READY', duration: 0}
];

function showProcessingSequence(callback) {
  const overlay = document.createElement('div');
  overlay.className = 'processing-overlay';
  overlay.style.cssText = `
    position:fixed;top:0;left:0;width:100vw;height:100vh;
    background:rgba(0,0,0,0.85);color:#fff;display:flex;
    flex-direction:column;align-items:center;justify-content:center;
    font-family:var(--font-family);z-index:1000;`;
  const labelEl = document.createElement('div');
  labelEl.style.fontSize = '1.5rem';
  overlay.appendChild(labelEl);
  document.body.appendChild(overlay);

  let i = 0;
  function next() {
    if (i >= stages.length) {
      document.body.removeChild(overlay);
      callback();
      return;
    }
    const stage = stages[i];
    labelEl.textContent = stage.label;
    gsap.fromTo(labelEl, {opacity:0, y:20}, {opacity:1, y:0, duration:0.4});
    setTimeout(() => {
      gsap.to(labelEl, {opacity:0, y:-20, duration:0.3, onComplete: () => {
        i++;
        next();
      }});
    }, stage.duration);
  }
  next();
}

// ---------- Panel creation ----------
function createPanel(title, contentEl) {
  const panel = document.createElement('div');
  panel.className = 'panel';
  const header = document.createElement('div');
  header.className = 'panel-header';
  header.textContent = title;
  const body = document.createElement('div');
  body.className = 'panel-body';
  if (contentEl) body.appendChild(contentEl);
  panel.appendChild(header);
  panel.appendChild(body);
  return panel;
}

function renderPlots(result, fileName) {
  const panelsContainer = document.getElementById('panels');
  panelsContainer.innerHTML = '';

  // Waveform panel
  const waveDiv = document.createElement('div');
  waveDiv.id = 'waveform';
  waveDiv.style.height = '300px';
  const wavePanel = createPanel('Waveform', waveDiv);
  panelsContainer.appendChild(wavePanel);

  // Spectrum panel
  const specDiv = document.createElement('div');
  specDiv.id = 'spectrum';
  specDiv.style.height = '300px';
  const specPanel = createPanel('Spectrum', specDiv);
  panelsContainer.appendChild(specPanel);

  // Spectrogram panel (conditional)
  if (result.spectrogram) {
    const specgDiv = document.createElement('div');
    specgDiv.id = 'spectrogram';
    specgDiv.style.height = '300px';
    const specgPanel = createPanel('Spectrogram', specgDiv);
    panelsContainer.appendChild(specgPanel);
  }

  // Constellation (for IQ) panel
  if (result.constellation) {
    const constDiv = document.createElement('div');
    constDiv.id = 'constellation';
    constDiv.style.height = '300px';
    const constPanel = createPanel('Constellation', constDiv);
    panelsContainer.appendChild(constPanel);
  }

  // Parameters panel
  const paramDiv = document.createElement('pre');
  paramDiv.style.whiteSpace = 'pre-wrap';
  paramDiv.textContent = JSON.stringify(result.parameters || {}, null, 2);
  const paramPanel = createPanel('Extracted Parameters', paramDiv);
  panelsContainer.appendChild(paramPanel);

  // Plotly charts – use the data returned by the backend if present
  // The backend currently returns a full result dict; we assume it contains
  // arrays for waveform, spectrum, etc. (adapt as needed).
  // Waveform
  if (result.waveform) {
    Plotly.newPlot('waveform', [{
      x: result.waveform.time,
      y: result.waveform.amplitude,
      type: 'scatter',
      mode: 'lines',
      line: {color: 'var(--color-signal)'}
    }], {margin:{l:40,r:20,t:30,b:40},title:''});
  }
  // Spectrum
  if (result.spectrum) {
    Plotly.newPlot('spectrum', [{
      x: result.spectrum.frequency,
      y: result.spectrum.power,
      type: 'scatter',
      mode: 'lines',
      line: {color: 'var(--color-accent)'}
    }], {margin:{l:40,r:20,t:30,b:40},title:''});
  }
  // Spectrogram
  if (result.spectrogram) {
    Plotly.newPlot('spectrogram', [{
      x: result.spectrogram.time,
      y: result.spectrogram.frequency,
      z: result.spectrogram.power,
      type: 'heatmap',
      colorscale: [[0, '#0D1117'], [1, 'var(--color-signal)']]
    }], {margin:{l:40,r:20,t:30,b:40},title:''});
  }
  // Constellation
  if (result.constellation) {
    Plotly.newPlot('constellation', [{
      x: result.constellation.I,
      y: result.constellation.Q,
      mode: 'markers',
      type: 'scatter',
      marker: {color: 'var(--color-signal)', size:4}
    }], {margin:{l:40,r:20,t:30,b:40},title:'',xaxis:{title:'I'},yaxis:{title:'Q'}});
  }
}

// ---------- File handling ----------
async function handleFile(file) {
  setStatus(`Uploading ${file.name}…`);
  const form = new FormData();
  form.append('file', file);
  try {
    // Show animation sequence first
    showProcessingSequence(async () => {
      // After animation, actually send to backend
      const response = await fetch('/api/signals', {
        method: 'POST',
        body: form
      });
      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();
      setStatus('Analysis complete', 'var(--color-success)');
      // Hide landing hero, show workspace
      document.querySelector('.hero').style.display = 'none';
      workspace.style.display = 'block';
      // Render panels with result data
      renderPlots(data.result, file.name);
    });
  } catch (err) {
    console.error(err);
    setStatus('Error: ' + err.message, 'var(--color-error)');
    alert('Upload or analysis failed: ' + err.message);
  }
}

// ---------- Keyboard shortcut for command surface (placeholder) ----------
window.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    alert('Command surface not implemented yet – placeholder');
  }
});
