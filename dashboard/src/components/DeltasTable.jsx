const RETRIEVAL_METRICS = [
  ['recall_at_5', 'recall@5'],
  ['recall_at_10', 'recall@10'],
  ['mrr', 'MRR'],
  ['ndcg_at_10', 'nDCG@10'],
]

function fmt(x) {
  return `${x >= 0 ? '+' : ''}${x.toFixed(3)}`
}

export default function DeltasTable({ deltas }) {
  if (!deltas || Object.keys(deltas).length === 0) return null

  return (
    <table className="deltas-table">
      <thead>
        <tr>
          <th>Stage pair</th>
          {RETRIEVAL_METRICS.map(([, label]) => (
            <th key={label}>{label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {Object.entries(deltas).map(([pair, metrics]) => (
          <tr key={pair}>
            <td>{pair.replace('->', ' → ')}</td>
            {RETRIEVAL_METRICS.map(([id]) => {
              const stats = metrics[id]
              const text = `${fmt(stats.delta)} [${fmt(stats.ci_lower)}, ${fmt(stats.ci_upper)}]`
              return (
                <td key={id} className={stats.significant ? 'delta-significant' : ''}>
                  {stats.significant ? <strong>{text}</strong> : text}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
