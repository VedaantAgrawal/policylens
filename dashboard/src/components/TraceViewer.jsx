import { renderBold } from '../markdown'

const TOOL_LABELS = {
  search_corpus: 'search_corpus',
  fetch_section: 'fetch_section',
  compare_provisions: 'compare_provisions',
  extract_numeric_field: 'extract_numeric_field',
}

function formatInput(input) {
  return Object.entries(input)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`)
    .join(', ')
}

export default function TraceViewer({ trace }) {
  if (!trace || trace.length === 0) return null

  return (
    <details className="trace-viewer">
      <summary>
        Agent trace ({trace.filter((e) => e.type === 'tool_call').length} tool call
        {trace.filter((e) => e.type === 'tool_call').length === 1 ? '' : 's'})
      </summary>
      <ol className="trace-list">
        {trace.map((event, i) => {
          if (event.type === 'assistant_text') {
            return (
              <li key={i} className="trace-event trace-event--text">
                <span className="trace-badge trace-badge--think">thinking</span>
                <span className="trace-text">{renderBold(event.text)}</span>
              </li>
            )
          }
          if (event.type === 'tool_call') {
            return (
              <li key={i} className="trace-event trace-event--tool">
                <span className="trace-badge trace-badge--tool">
                  {TOOL_LABELS[event.tool] || event.tool}
                </span>
                <span className="trace-text">{formatInput(event.input)}</span>
              </li>
            )
          }
          return null
        })}
      </ol>
    </details>
  )
}
