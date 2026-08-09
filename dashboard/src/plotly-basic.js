// plotly.js-basic-dist-min keeps the bundle small (we only need bar charts
// with error bars) — react-plotly.js's default export expects the full
// plotly.js package, so it's wired up via the documented factory pattern
// instead of the plain `import Plot from 'react-plotly.js'` import.
import Plotly from 'plotly.js-basic-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'

const Plot = createPlotlyComponent(Plotly)
export default Plot
