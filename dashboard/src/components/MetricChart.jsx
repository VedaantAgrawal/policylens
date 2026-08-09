import Plot from '../plotly-basic'
import { usePalette } from '../theme'

const STAGE_COLORS = ['series1', 'series2', 'series3', 'series4']

/**
 * One metric, one bar per stage, 95% CI as an error bar.
 * `stages`: [[id, label], ...]. `data`: { [stageId]: {mean, ci_lower, ci_upper, n?} }
 */
export default function MetricChart({ title, stages, data }) {
  const palette = usePalette()

  const labels = stages.map(([, label]) => label)
  const means = stages.map(([id]) => data[id]?.mean ?? null)
  const errPlus = stages.map(([id]) => (data[id] ? data[id].ci_upper - data[id].mean : 0))
  const errMinus = stages.map(([id]) => (data[id] ? data[id].mean - data[id].ci_lower : 0))
  const colors = stages.map((_, i) => palette[STAGE_COLORS[i % STAGE_COLORS.length]])

  return (
    <div className="metric-chart">
      <Plot
        data={[
          {
            type: 'bar',
            x: labels,
            y: means,
            marker: { color: colors },
            error_y: { type: 'data', array: errPlus, arrayminus: errMinus, color: palette.textSecondary, thickness: 1.5, width: 4 },
            hovertemplate: '%{x}: %{y:.3f}<extra></extra>',
          },
        ]}
        layout={{
          title: { text: title, font: { size: 14, color: palette.textPrimary } },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: palette.textSecondary, size: 11, family: 'system-ui, -apple-system, sans-serif' },
          margin: { t: 36, r: 12, b: 64, l: 40 },
          yaxis: { gridcolor: palette.gridline, zerolinecolor: palette.gridline, rangemode: 'tozero' },
          xaxis: { gridcolor: palette.gridline, tickangle: -30, automargin: true },
          showlegend: false,
          height: 240,
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  )
}
