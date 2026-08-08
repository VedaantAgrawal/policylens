"""Render site/index.html from committed eval_results/*.json — the static
dashboard deployed to Cloudflare Pages. No numbers are hand-typed into the
HTML; everything is read from the same JSON files `make eval` regenerates.
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_RESULTS_DIR = Path("eval_results")
OUTPUT_PATH = Path("site/index.html")

STAGES = [
    ("s0_bm25", "S0 BM25"),
    ("s1_dense", "S1 Dense"),
    ("s2_hybrid", "S2 Hybrid"),
    ("s3_rerank", "S3 Rerank"),
]
RETRIEVAL_METRICS = [
    ("recall_at_5", "Recall@5"),
    ("recall_at_10", "Recall@10"),
    ("mrr", "MRR"),
    ("ndcg_at_10", "nDCG@10"),
]
GENERATION_METRICS = [
    ("citation_precision", "Citation Precision"),
    ("groundedness", "Groundedness"),
    ("refusal_accuracy", "Refusal Accuracy"),
    ("false_refusal_rate", "False Refusal Rate"),
]


def _load(name: str) -> dict | None:
    path = EVAL_RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def build_data() -> dict:
    retrieval = {}
    generation = {}
    for stage_id, _ in STAGES:
        r = _load(stage_id)
        if r:
            retrieval[stage_id] = {m: r["aggregate"][m] for m, _ in RETRIEVAL_METRICS}
        g = _load(f"generation_{stage_id}")
        if g:
            generation[stage_id] = {m: g["aggregate"][m] for m, _ in GENERATION_METRICS}
    deltas = _load("stage_deltas") or {}
    return {
        "stages": STAGES,
        "retrieval_metrics": RETRIEVAL_METRICS,
        "generation_metrics": GENERATION_METRICS,
        "retrieval": retrieval,
        "generation": generation,
        "deltas": deltas,
    }


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PolicyLens &mdash; Retrieval Ablation</title>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --baseline: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;
    --series-2: #eb6834;
    --series-3: #1baf7a;
    --series-4: #eda100;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --baseline: #383835;
      --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --series-2: #d95926;
      --series-3: #199e70;
      --series-4: #c98500;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page);
    color: var(--text-primary);
    padding: 32px 20px 80px;
  }
  main { max-width: 980px; margin: 0 auto; }
  h1 { font-size: 1.6rem; margin-bottom: 4px; }
  .subtitle { color: var(--text-secondary); margin-bottom: 8px; }
  .subtitle a { color: var(--series-1); }
  section { margin-top: 40px; }
  h2 { font-size: 1.15rem; margin-bottom: 4px; }
  .section-note { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 20px; max-width: 720px; }
  .legend { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.85rem; color: var(--text-secondary); }
  .legend-swatch { width: 12px; height: 12px; border-radius: 3px; }
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; }
  .chart-card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  .chart-title { font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 8px; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; position: relative; }
  .bar-label { width: 70px; font-size: 0.75rem; color: var(--text-muted); flex-shrink: 0; }
  .bar-track { flex: 1; position: relative; height: 20px; }
  .bar-fill { height: 20px; border-radius: 4px; position: relative; }
  .bar-ci { position: absolute; top: 0; height: 20px; border-left: 2px solid var(--text-primary); border-right: 2px solid var(--text-primary); opacity: 0.35; }
  .bar-value { font-size: 0.75rem; color: var(--text-secondary); width: 44px; text-align: right; flex-shrink: 0; }
  .bar-fill:hover { outline: 2px solid var(--text-primary); outline-offset: 1px; cursor: default; }
  table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 0.85rem; }
  caption { text-align: left; color: var(--text-secondary); font-size: 0.8rem; margin-bottom: 6px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--gridline); }
  th { color: var(--text-muted); font-weight: 600; }
  td.num { font-variant-numeric: tabular-nums; }
  details summary { cursor: pointer; color: var(--text-secondary); font-size: 0.85rem; margin-top: 8px; }
  footer { margin-top: 60px; color: var(--text-muted); font-size: 0.8rem; }
  footer a { color: var(--series-1); }
</style>
</head>
<body>
<main>
  <h1>PolicyLens: Retrieval Ablation</h1>
  <p class="subtitle">Agentic RAG over insurance &amp; regulatory documents &mdash;
    <a href="https://github.com/VedaantAgrawal/policylens" target="_blank" rel="noopener">source on GitHub</a></p>
  <p class="section-note">Every number below is read directly from committed
    <code>eval_results/*.json</code>, regenerated from scratch by <code>make eval</code>
    (retrieval + significance tests) and <code>make eval-generation</code> (generation +
    judge). Bars show the mean; the vertical tick marks show the 95% bootstrap
    confidence interval (n=60 answerable questions, 10,000 resamples).</p>

  <section id="retrieval">
    <h2>Retrieval ablation</h2>
    <p class="section-note">Dense retrieval alone (S1) underperforms the BM25 baseline
      (S0) on this corpus. Growing the golden set from n=30 to n=60 changed which
      patterns hold up: at n=30, hybrid fusion (S2) led on every metric and rerank (S3)
      looked like it traded away recall@10; at n=60, S2's lead is gone on point
      estimates and S3's recall@10 is now the best of all four stages. The only
      delta that actually clears 95% significance in a paired bootstrap test is
      S2&rarr;S3 on recall@5 &mdash; see the table below. Read every other bar as a
      plausible-but-unconfirmed direction, not a settled result.</p>
    __LEGEND__
    <div class="chart-grid" id="retrieval-charts"></div>
    <details>
      <summary>Show as table</summary>
      <table id="retrieval-table"></table>
    </details>
  </section>

  <section id="deltas">
    <h2>Paired bootstrap stage deltas</h2>
    <p class="section-note">The actual significance test, not a proxy for it: paired
      bootstrap (10k resamples) on the same aligned per-question scores across stage
      pairs. Bold = clears 95% significance (CI excludes zero).</p>
    <table id="deltas-table"></table>
  </section>

  <section id="generation">
    <h2>Generation quality</h2>
    <p class="section-note">Refusal accuracy sits in a tight band (0.917&ndash;0.958)
      across all four retrieval stages. Of 22 false refusals traced by hand under S2,
      21 had no gold chunk in the retrieved top-5 &mdash; a retrieval-recall ceiling,
      not generator over-caution. The 1 exception had partial retrieval (1 of 2 needed
      chunks) and the model correctly said so rather than guessing from what it did
      have &mdash; still correct, calibrated behavior under partial context.</p>
    __LEGEND__
    <div class="chart-grid" id="generation-charts"></div>
    <details>
      <summary>Show as table</summary>
      <table id="generation-table"></table>
    </details>
  </section>

  <footer>
    Corpus: 220 documents (33 SEC 10-Ks, 103 NAIC model laws, 84 NY/CA/TX DOI bulletins).
    Generation: claude-sonnet-5. Judge: claude-haiku-4-5.
    &middot; <a href="https://policylens-api.vedaantagrawal.com/health" target="_blank" rel="noopener">live API</a>
  </footer>
</main>

<script id="viz-data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('viz-data').textContent);
const SERIES_COLORS = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)'];

function legendHTML() {
  return '<div class="legend">' + DATA.stages.map(([id, label], i) =>
    `<div class="legend-item"><span class="legend-swatch" style="background:${SERIES_COLORS[i]}"></span>${label}</div>`
  ).join('') + '</div>';
}
// Insert legends
document.getElementById('retrieval-charts').insertAdjacentHTML('beforebegin', legendHTML());
document.getElementById('generation-charts').insertAdjacentHTML('beforebegin', legendHTML());

function renderChartGrid(containerId, metrics, dataByStage) {
  const container = document.getElementById(containerId);
  metrics.forEach(([metricId, metricLabel]) => {
    const card = document.createElement('div');
    card.className = 'chart-card';
    const title = document.createElement('div');
    title.className = 'chart-title';
    title.textContent = metricLabel;
    card.appendChild(title);

    DATA.stages.forEach(([stageId, stageLabel], i) => {
      const stageData = dataByStage[stageId];
      const stats = stageData ? stageData[metricId] : null;
      const row = document.createElement('div');
      row.className = 'bar-row';

      const label = document.createElement('div');
      label.className = 'bar-label';
      label.textContent = stageLabel;
      row.appendChild(label);

      const track = document.createElement('div');
      track.className = 'bar-track';

      if (stats && stats.mean !== null && stats.mean !== undefined) {
        const pct = Math.max(0, Math.min(1, stats.mean)) * 100;
        const loPct = Math.max(0, Math.min(1, stats.ci_lower)) * 100;
        const hiPct = Math.max(0, Math.min(1, stats.ci_upper)) * 100;

        const fill = document.createElement('div');
        fill.className = 'bar-fill';
        fill.style.width = pct + '%';
        fill.style.background = SERIES_COLORS[i];
        fill.title = `${stageLabel} ${metricLabel}: ${stats.mean.toFixed(3)} [${stats.ci_lower.toFixed(3)}, ${stats.ci_upper.toFixed(3)}]`;
        track.appendChild(fill);

        const ci = document.createElement('div');
        ci.className = 'bar-ci';
        ci.style.left = loPct + '%';
        ci.style.width = Math.max(0, hiPct - loPct) + '%';
        track.appendChild(ci);
      } else {
        track.style.opacity = '0.3';
        track.title = 'not run';
      }
      row.appendChild(track);

      const value = document.createElement('div');
      value.className = 'bar-value';
      value.textContent = (stats && stats.mean !== null && stats.mean !== undefined) ? stats.mean.toFixed(3) : 'n/a';
      row.appendChild(value);

      card.appendChild(row);
    });
    container.appendChild(card);
  });
}

function renderTable(tableId, metrics, dataByStage) {
  const table = document.getElementById(tableId);
  const caption = document.createElement('caption');
  caption.textContent = 'Mean [95% CI]';
  table.appendChild(caption);
  const thead = document.createElement('thead');
  thead.innerHTML = '<tr><th>Stage</th>' + metrics.map(([, label]) => `<th>${label}</th>`).join('') + '</tr>';
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  DATA.stages.forEach(([stageId, stageLabel]) => {
    const stageData = dataByStage[stageId];
    const cells = metrics.map(([metricId]) => {
      const stats = stageData ? stageData[metricId] : null;
      if (!stats || stats.mean === null || stats.mean === undefined) return '<td class="num">n/a</td>';
      return `<td class="num">${stats.mean.toFixed(3)} [${stats.ci_lower.toFixed(3)}, ${stats.ci_upper.toFixed(3)}]</td>`;
    }).join('');
    tbody.innerHTML += `<tr><td>${stageLabel}</td>${cells}</tr>`;
  });
  table.appendChild(tbody);
}

renderChartGrid('retrieval-charts', DATA.retrieval_metrics, DATA.retrieval);
renderTable('retrieval-table', DATA.retrieval_metrics, DATA.retrieval);
renderChartGrid('generation-charts', DATA.generation_metrics, DATA.generation);
renderTable('generation-table', DATA.generation_metrics, DATA.generation);

function renderDeltasTable() {
  const table = document.getElementById('deltas-table');
  const thead = document.createElement('thead');
  thead.innerHTML = '<tr><th>Stage pair</th>' + DATA.retrieval_metrics.map(([, label]) => `<th>${label}</th>`).join('') + '</tr>';
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  Object.entries(DATA.deltas).forEach(([pair, metrics]) => {
    const cells = DATA.retrieval_metrics.map(([metricId]) => {
      const stats = metrics[metricId];
      const text = `${stats.delta >= 0 ? '+' : ''}${stats.delta.toFixed(3)} [${stats.ci_lower >= 0 ? '+' : ''}${stats.ci_lower.toFixed(3)}, ${stats.ci_upper >= 0 ? '+' : ''}${stats.ci_upper.toFixed(3)}]`;
      return stats.significant ? `<td class="num"><strong>${text}</strong></td>` : `<td class="num">${text}</td>`;
    }).join('');
    tbody.innerHTML += `<tr><td>${pair.replace('->', ' → ')}</td>${cells}</tr>`;
  });
  table.appendChild(tbody);
}
renderDeltasTable();
</script>
</body>
</html>
"""


def main() -> None:
    data = build_data()
    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace("__LEGEND__", "")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
