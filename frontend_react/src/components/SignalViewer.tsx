import React from 'react';
import Plot from './PlotComponent';

interface Props {
  data: any;
}

export const SignalViewer: React.FC<Props> = ({ data }) => {
  if (!data) {
    return <p>No analysis data available.</p>;
  }

  // Simple placeholder: show JSON and a basic waveform if available
  const waveform = data?.signal?.waveform || [];
  const spectrogram = data?.signal?.spectrogram || [];

  return (
    <div className="signal-viewer">
      <h3>Analysis Result</h3>
      <pre style={{ maxHeight: '200px', overflowY: 'auto' }}>{JSON.stringify(data, null, 2)}</pre>
      {waveform.length > 0 && (
        <Plot
          data={[{ y: waveform, type: 'scatter', mode: 'lines', name: 'Waveform' }]}
          layout={{ title: 'Waveform', autosize: true }}
        />
      )}
      {spectrogram.length > 0 && (
        <Plot
          data={[{ z: spectrogram, type: 'heatmap', name: 'Spectrogram' }]}
          layout={{ title: 'Spectrogram', autosize: true }}
        />
      )}
    </div>
  );
};
