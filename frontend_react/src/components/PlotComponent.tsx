import Plotly from 'plotly.js-dist-min';
import _createPlotlyComponent from 'react-plotly.js/factory';

const factory =
  typeof _createPlotlyComponent === 'function'
    ? _createPlotlyComponent
    : (_createPlotlyComponent as { default?: typeof _createPlotlyComponent })?.default;

const Plot = factory ? factory(Plotly) : () => null;
export default Plot;
