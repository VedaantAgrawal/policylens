import { useEffect, useState } from 'react'
import { fetchEvalResults } from '../api'
import MetricChart from './MetricChart'
import DeltasTable from './DeltasTable'

const STAGES = [
  ['s0_bm25', 'S0 BM25'],
  ['s1_dense', 'S1 Dense'],
  ['s2_hybrid', 'S2 Hybrid'],
  ['s3_rerank', 'S3 Rerank'],
]
const RETRIEVAL_METRICS = [
  ['recall_at_5', 'Recall@5'],
  ['recall_at_10', 'Recall@10'],
  ['mrr', 'MRR'],
  ['ndcg_at_10', 'nDCG@10'],
]
const GENERATION_METRICS = [
  ['citation_precision', 'Citation Precision'],
  ['groundedness', 'Groundedness'],
  ['refusal_accuracy', 'Refusal Accuracy'],
  ['false_refusal_rate', 'False Refusal Rate'],
]

export default function EvalView() {
  const [raw, setRaw] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchEvalResults().then(setRaw).catch((err) => setError(err.message))
  }, [])

  if (error) {
    return (
      <div className="eval-view eval-view--error">
        <p>Couldn't load eval results from the live API ({error}).</p>
        <p>
          The raw files are always available in{' '}
          <a href="https://github.com/VedaantAgrawal/policylens/tree/main/eval_results" target="_blank" rel="noopener noreferrer">
            eval_results/
          </a>{' '}
          on GitHub.
        </p>
      </div>
    )
  }
  if (!raw) return <div className="eval-view">Loading eval results…</div>

  const retrieval = Object.fromEntries(
    STAGES.map(([id]) => [id, raw[id] ? Object.fromEntries(RETRIEVAL_METRICS.map(([m]) => [m, raw[id].aggregate[m]])) : {}]),
  )
  const generation = Object.fromEntries(
    STAGES.map(([id]) => [
      id,
      raw[`generation_${id}`]
        ? Object.fromEntries(GENERATION_METRICS.map(([m]) => [m, raw[`generation_${id}`].aggregate[m]]))
        : {},
    ]),
  )
  const latency = raw.latency_cost_s2_hybrid

  return (
    <div className="eval-view">
      <p className="section-note">
        Every number here is fetched live from <code>{'{API}'}/eval</code>, which serves the same committed{' '}
        <code>eval_results/*.json</code> files <code>make eval</code> regenerates — nothing on this page is hand-typed.
      </p>

      <section>
        <h2>Retrieval ablation</h2>
        <p className="section-note">
          Growing the golden set from n=30 to n=60 reversed two patterns that looked solid at the smaller sample — see
          the significance table below for what actually survives a paired bootstrap test.
        </p>
        <div className="chart-grid">
          {RETRIEVAL_METRICS.map(([id, label]) => (
            <MetricChart key={id} title={label} stages={STAGES} data={Object.fromEntries(STAGES.map(([sid]) => [sid, retrieval[sid][id]]))} />
          ))}
        </div>
      </section>

      <section>
        <h2>Paired bootstrap stage deltas</h2>
        <p className="section-note">
          The actual significance test — 10k-resample paired bootstrap on aligned per-question scores, not CI overlap.
          <strong> Bold</strong> = clears 95% significance.
        </p>
        <DeltasTable deltas={raw.stage_deltas} />
      </section>

      <section>
        <h2>Generation quality</h2>
        <div className="chart-grid">
          {GENERATION_METRICS.map(([id, label]) => (
            <MetricChart
              key={id}
              title={label}
              stages={STAGES}
              data={Object.fromEntries(STAGES.map(([sid]) => [sid, generation[sid][id]]))}
            />
          ))}
        </div>
      </section>

      {latency && (
        <section>
          <h2>Ops: latency and cost (S2 hybrid)</h2>
          <div className="stat-row">
            <div className="stat-tile">
              <div className="stat-value">{latency.latency_seconds.p50.toFixed(2)}s</div>
              <div className="stat-label">p50 latency</div>
            </div>
            <div className="stat-tile">
              <div className="stat-value">{latency.latency_seconds.p95.toFixed(2)}s</div>
              <div className="stat-label">p95 latency</div>
            </div>
            <div className="stat-tile">
              <div className="stat-value">${latency.cost_usd.mean_per_query.toFixed(4)}</div>
              <div className="stat-label">mean cost/query</div>
            </div>
            <div className="stat-tile">
              <div className="stat-value">${latency.cost_usd.total.toFixed(2)}</div>
              <div className="stat-label">total cost (n={latency.num_queries})</div>
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
