# PolicyLens

Agentic RAG over public insurance and regulatory documents (SEC EDGAR life-insurer
10-Ks, NAIC model laws, state Department of Insurance bulletins) — built around a
measured retrieval ablation, not a chat demo.

Every number below comes from a committed file in [`eval_results/`](eval_results/),
regenerated from scratch by `make eval` (retrieval + significance tests, free) and
`make eval-generation` (generation + judge, requires an Anthropic API key, ~$2-3 in
API spend). Nothing here is hand-typed.

**Live:** [interactive dashboard](https://policylens.vedaantagrawal.com) (React/Vite/Plotly,
chat with the agent + browse the ablation, Cloudflare Pages) ·
[API](https://policylens-api.vedaantagrawal.com/health) (FastAPI behind a Cloudflare Tunnel to a local Docker container)

## Results

### Retrieval ablation (recall@k, MRR, nDCG@10 — 95% bootstrap CI, n=60 answerable questions)

| Stage | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| **S0** BM25 (baseline) | 0.617 [0.500, 0.733] | 0.700 [0.583, 0.817] | 0.467 [0.358, 0.578] | 0.521 [0.416, 0.625] |
| **S1** dense (all-MiniLM-L6-v2) | 0.567 [0.442, 0.692] | 0.592 [0.467, 0.717] | 0.440 [0.332, 0.551] | 0.470 [0.365, 0.578] |
| **S2** hybrid (RRF, S0+S1) | 0.558 [0.433, 0.683] | 0.683 [0.567, 0.800] | 0.489 [0.377, 0.601] | 0.532 [0.425, 0.639] |
| **S3** cross-encoder rerank (ms-marco-MiniLM-L-6-v2, over S2's top-20) | 0.650 [0.533, 0.767] | 0.758 [0.650, 0.867] | 0.515 [0.409, 0.621] | 0.572 [0.471, 0.671] |

### Paired bootstrap stage deltas (95% CI, 10k resamples, aligned by question)

Per-stage confidence intervals above tell you the uncertainty in each stage's own
mean — they do **not** tell you whether one stage significantly beats another, since
overlapping CIs can still hide a real paired difference. This table is the actual
test, not a proxy for it:

| Stage pair | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| S0 → S1 | −0.050 [−0.183, +0.083] | −0.108 [−0.225, +0.008] | −0.028 [−0.138, +0.087] | −0.051 [−0.151, +0.051] |
| S0 → S2 | −0.058 [−0.175, +0.058] | −0.017 [−0.100, +0.067] | +0.021 [−0.044, +0.087] | +0.011 [−0.053, +0.074] |
| **S2 → S3** | **+0.092 [+0.008, +0.183]** | +0.075 [+0.000, +0.158] | +0.026 [−0.044, +0.104] | +0.040 [−0.022, +0.108] |
| S0 → S3 | +0.033 [−0.050, +0.117] | +0.058 [−0.009, +0.133] | +0.047 [−0.022, +0.122] | +0.051 [−0.007, +0.113] |

**Exactly one delta in this entire table is statistically significant: S2→S3 on
recall@5.** Everything else — including S0→S2, the "hybrid beats baseline" story —
does not clear 95% significance at this sample size.

**Growing the golden set from n=30 to n=60 changed conclusions, not just narrowed
error bars — and that's the point of doing this properly.** At n=30, S2 hybrid led
BM25 on every retrieval metric and S3 rerank *lost* ground on recall@10 relative to
S2 (0.767 → 0.700), which read as "reranking trades recall for precision." Neither
survives at n=60: S2's point estimate now trails S0 on recall@5, and S3's recall@10
(0.758) is the *best* of all four stages, not the worst. Both of those n=30 patterns
were noise from a sample too small to trust — which is exactly why this project
insists on a paired significance test (`eval/stage_deltas.py`) instead of eyeballing
point estimates or CI overlap. A golden set of 60 answerable questions is still small
by production standards; treat the one surviving significant result (S2→S3 on
recall@5) as the only retrieval claim this ablation currently supports with
confidence, and everything else as a plausible-but-unconfirmed direction.

### Generation quality (citation precision, groundedness, refusal accuracy — 95% bootstrap CI, n=60/24)

| Retriever used for generation | citation precision | groundedness | refusal accuracy | false refusal rate |
|---|---|---|---|---|
| S0 BM25 | 1.000 [1.000, 1.000] (n=51) | 0.977 [0.930, 1.000] (n=43) | 0.958 [0.875, 1.000] (n=24) | 0.283 [0.167, 0.400] (n=60) |
| S1 dense | 1.000 [1.000, 1.000] (n=46) | 0.976 [0.927, 1.000] (n=41) | 0.917 [0.792, 1.000] (n=24) | 0.317 [0.200, 0.433] (n=60) |
| S2 hybrid | 1.000 [1.000, 1.000] (n=49) | 0.974 [0.921, 1.000] (n=38) | 0.917 [0.792, 1.000] (n=24) | 0.367 [0.250, 0.483] (n=60) |
| S3 rerank | 1.000 [1.000, 1.000] (n=52) | 0.936 [0.851, 1.000] (n=47) | 0.917 [0.792, 1.000] (n=24) | 0.217 [0.117, 0.333] (n=60) |

Generation runs on `claude-sonnet-5` with a structured JSON output (answer + inline
`[chunk_id]` citations + a self-reported `answerable` flag); groundedness is judged by
`claude-haiku-4-5` against a written claim-by-claim rubric; citation precision is a
deterministic check (cited chunk actually in the retrieved set), not an LLM judgment.

Refusal accuracy sits in a tight 0.917–0.958 band across all four stages (22-23 of 24
unanswerable questions correctly refused), and groundedness/false-refusal differences
between stages mostly sit inside each other's confidence intervals. S3's groundedness
(0.936) is the one value worth watching — lower than the other three and closest to
its own CI floor — but still well within noise at this n; not a claim this sample size
can confirm or rule out.

**Most false refusals are a retrieval problem, not a generation problem — but not
all of them, and that nuance only showed up after growing the sample.** I traced
every false refusal under S2 by hand: of 22 answerable questions the model refused,
21 had no gold chunk in the retrieved top-5 — the generator was never handed the
right context and correctly declined to guess. The 1 exception (q025, about a TDI
storm-relief bulletin) is more interesting than a clean miss: retrieval found *one*
of the two gold chunks, and the model correctly recognized that the retrieved portion
didn't contain the specific FEMA-assistance detail the question asked about, and said
so explicitly rather than filling the gap from the adjacent chunk it did have. That's
correct, well-calibrated behavior under partial context, not a failure — but it means
the honest claim is "false refusals are overwhelmingly retrieval-driven" (21/22), not
"generation never causes a false refusal" (which held, barely, at the old n=30 by
coincidence of which 9 questions happened to be sampled).

### Ops: latency and cost (S2 hybrid, n=60, `claude-sonnet-5`)

| p50 latency | p95 latency | mean latency | mean cost/query | total cost |
|---|---|---|---|---|
| 3.85s | 7.01s | 4.17s (retrieval 0.28s + generation 3.89s) | $0.0127 | $0.76 |

Retrieval is not the bottleneck — generation is ~93% of end-to-end latency, and
retrieval latency roughly doubled (0.11s → 0.28s) after the corpus grew from ~6,500 to
~24,500 chunks — reasonable for two local-index scans (BM25 + dense) whose cost scales
with corpus size. Reproduce with `make eval-latency-cost` (requires
`ANTHROPIC_API_KEY`, makes real API calls). Every `/query` response also carries an
`X-Response-Time-Ms` header and a `cost_usd` field, computed the same way, so this
isn't a number that only exists in a committed JSON file — you can watch it on the
live API.

## Architecture

```
data/manifest.jsonl (committed)          data/golden/golden_questions.json (committed)
        │                                            │
        ▼                                            │
  ingest/fetch.py → data/raw/ (gitignored)            │
        │                                             │
        ▼                                             │
  ingest/chunk.py → data/processed/chunks.jsonl        │
        │           (section-aware: heading + page)    │
        ▼                                             │
  retrieval/{bm25,dense,hybrid,rerank}.py  ◄───────────────┘
        │                                    eval/runner.py
        ▼                                    (recall@k, MRR, nDCG@10,
  generation/generate.py                      bootstrap CI)
  (cited synthesis, refusal path)                    │
        │                                            ▼
        ▼                                    eval_results/*.json (committed)
  eval/judge.py (groundedness)                        │
        │                                             ▼
        ▼                                    eval/ablation.py → this README
  agent/tools.py (4 tools) → agent/loop.py
  (tool_runner, structured trace)
        │
        ▼
  serving/app.py (FastAPI: /query, /agent/query, /eval, /health)
        │
        ▼
  dashboard/ (React/Vite/Plotly: chat view + eval view)
```

- **Corpus** (220 docs, ~24,500 chunks — grown from an original 109/~6,500 pass):
  33 SEC 10-Ks (last 3 annual filings each, where available, from 11 publicly
  traded life insurers — MetLife, Prudential, Lincoln National, Brighthouse,
  Corebridge, Unum, Principal Financial, Globe Life, Voya, Genworth, F&G
  Annuities & Life; mutual companies like MassMutual still can't appear here,
  they don't file 10-Ks), 103 NAIC model laws (life/annuity/reinsurance/solvency
  plus a second wave covering producer licensing, market conduct, investments,
  and receivership — all confirmed free at content.naic.org, deliberately
  excluding Long-Term Care #640/641 and Medicare Supplement #650/651/660, which
  the golden set uses as deliberate-exclusion trap questions), 84 state DOI/DFS
  bulletins (NY, CA, TX — Massachusetts was dropped after mass.gov returned 403
  to automated requests; that's a WAF, not a robots.txt disallow, and bypassing
  it isn't "official API/bulk data" access). Still short of the original
  300–800 target — see Known limitations.
- **Chunking** is section-aware: PDF page boundaries and HTML `<h1-6>` tags plus a
  line-shape heuristic (`Item 7.`, `Section 3.`, `ARTICLE I`, all-caps headers) drive
  chunk boundaries, so `section_heading` and `page` metadata on each chunk are
  accurate to what's actually in it — that's what inline citations point back to.
- **Golden set** (84 questions: 60 answerable + 24 unanswerable, grown from an
  original 42) is hand-written against real sampled chunk text, not LLM-generated —
  every `relevant_chunk_id` is checked at load time (`golden_set.py`) against the
  actual current corpus and rejected if it doesn't exist, plus each new question's
  claimed fact was spot-verified with an automated substring check against its cited
  chunk text before being committed. The unanswerable subset covers distinct
  refusal-trap categories: entities not in corpus (mutual companies like MassMutual,
  New York Life, Guardian Life), jurisdictions not in corpus (MA/FL/WA/IL/PA/OH),
  NAIC model laws deliberately excluded from the curated list (Long-Term Care
  #640/641, Medicare Supplement #650/651/660 — to test against false-positive
  retrieval), wrong filing type or period (10-Q vs 10-K, quarterly vs annual),
  stale/future-dated or fabricated-sounding bulletins, out-of-domain questions
  (federal monetary policy, IRC tax code), double-miss traps (wrong entity *and*
  wrong jurisdiction in one question), and unverifiable operational precision claims.
- **Model-provider abstraction**: `providers/base.py` defines a one-method
  `ModelProvider` protocol; `AnthropicProvider` and `BedrockProvider` both implement
  it — verified live against a real AWS account, so generation/judge code never
  imports an SDK directly. Bedrock goes through `bedrock-runtime` (not the newer
  `bedrock-mantle` endpoint, which doesn't carry Anthropic models on this account)
  via a cross-region inference profile ID (`us.anthropic.claude-sonnet-4-6` by
  default — Sonnet 5 itself is listed as `ACTIVE` in `list_foundation_models` but
  returns a 403 "not available for this account," an AWS-side entitlement gap on
  new-model rollout rather than a config issue).
- **Prompt-injection defense**: retrieved chunk text is wrapped in `<source>` tags
  with an explicit system-prompt instruction to treat it as inert data, never as
  instructions — a chunk containing "ignore prior instructions and..." is
  attacker-controlled content the model must not act on.
- **PII redaction on ingest**: every chunk is scanned for SSN/email/phone/credit-card
  patterns before being written, replaced with `[REDACTED-<TYPE>]`. Not theoretical —
  it fires 153 times on the real corpus, almost entirely legitimate department contact
  info in state bulletins ("Consumer Hotline at [REDACTED-PHONE]"). The first pass of
  the phone/credit-card patterns was too loose and matched actuarial loss-development
  tables in the 10-Ks (columns of space-separated 3-4 digit numbers happen to be
  phone/card-shaped) — tightened to require parens or dashes, not bare spaces, and
  verified against zero overlap with golden-set-relevant chunks before trusting it.

## Reproduce it

```sh
git clone https://github.com/VedaantAgrawal/policylens.git
cd policylens
uv sync

make setup             # fetch corpus (220 docs, ~5 min, needs internet), chunk, embed
make test               # 89 unit tests, including eval metrics, significance tests,
                          #   golden-set validation, PII redaction, and agent tools
make eval               # retrieval ablation + paired significance tests — free, no API key
make eval-generation     # regenerate generation/groundedness/refusal — needs
                          #   ANTHROPIC_API_KEY in .env, costs ~$2-3 in API spend
make eval-latency-cost   # p50/p95 latency + cost-per-query for S2 — needs
                          #   ANTHROPIC_API_KEY, makes real API calls
uv run python scripts/spot_check_groundedness.py             # human review of a 20%
uv run python scripts/spot_check_groundedness.py --summarize  #   sample of judge calls
```

`make eval` alone reproduces every number in the retrieval ablation and significance
tables above from a clean clone with zero API cost — BM25, the dense embedding model
(`all-MiniLM-L6-v2`), and the cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) all run
locally. `make eval-generation` is separate and explicit about spending money, so the
free path stays free.

To run the API locally:

```sh
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
make serve               # FastAPI on :8000
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "How many days after a premium due date must a policyholder request a paid-up nonforfeiture benefit under the NAIC Standard Nonforfeiture Law?"}'
curl -X POST localhost:8000/agent/query -H "Content-Type: application/json" \
  -d '{"question": "Compare how MetLife and Prudential describe their reinsurance recoverables"}'
```

To run the dashboard against a local API (`cd dashboard && npm install && npm run dev`,
proxies to `localhost:8000` by default) — see [`dashboard/README.md`](dashboard/README.md).

## Known limitations

- **n=60 answerable questions is better than n=30, still small.** Growing the golden
  set (42→84 questions) turned up exactly one statistically significant retrieval
  delta (S2→S3 on recall@5) and reversed two point-estimate patterns that looked real
  at the smaller sample — see the paired-bootstrap section above. Treat the surviving
  significant result as the one confident claim this ablation supports, and everything
  else as a plausible-but-unconfirmed direction, not a settled result.
- **Corpus is 220 docs, not the original 300–800 target.** Grown from 109 (33 SEC
  10-Ks — 3 years each across 11 issuers, up from 1 year across 6 — plus 103 NAIC
  model laws and 84 state bulletins, up from 45/54), but still short of the floor.
  NY is capped at 14 bulletins by what's actually listed on its single circular-letter
  index page — CA and TX both hit the configured 35-per-state cap comfortably (CA's
  listing page alone has 164 available links). Getting NY higher would need either a
  URL-pattern-based fetch (its bulletin URLs follow a predictable
  `cl{year}-{number}` scheme) or discovering an archive/pagination endpoint, neither
  attempted yet.
- **Dense embeddings use a small, general-purpose model** (`all-MiniLM-L6-v2`, 384-dim,
  not domain-tuned), chosen to run locally at zero cost. A larger or
  insurance/legal-tuned embedding model would likely close or reverse the S0/S1 gap.
- **Chunking has a known artifact**: NAIC PDF page footers repeat the document title
  (e.g. "Insurance Data Security Model Law MO-668-2 (c) 2017...") on every page, which
  occasionally inflates lexical similarity for chunks adjacent to the correct one on
  title-mentioning queries. Documented in the S0 commit, not silently patched over.
- **Human spot-check of the groundedness judge exists as tooling, not yet as a
  completed review.** `scripts/spot_check_groundedness.py` samples 20% of judged
  generation records (currently 89 judged records → an 18-item sample, seed=42 for
  reproducibility) for a human to independently verdict against the same rubric the
  LLM judge used, then scores agreement with Cohen's kappa. Run it and `--summarize`
  to actually close this gap — the tool doesn't replace a human, it just makes the
  spot-check possible.
- **The agent layer (`/agent/query`) isn't covered by the ablation or significance
  tests above** — those measure single-shot `/query` retrieval+generation only. The
  agent's tool-selection quality (does it call the right tool, does it stop at the
  right turn) is verified by unit tests on `_build_tools()` and by live manual runs,
  not by a scored eval set the way retrieval and generation are.

## Agent layer

`/query` is single-shot retrieval + generation. `/agent/query` (and the dashboard's
chat view) runs a real multi-step agent instead, via the Anthropic SDK's
`client.beta.messages.tool_runner` — not a hand-rolled loop — over 4 tools defined in
`agent/tools.py` and wrapped with `@beta_tool` in `agent/loop.py`:

- `search_corpus(query, k)` — thin wrapper over `HybridRetriever.search`
- `fetch_section(doc_id, section_heading, page)` — pull a specific section once the
  agent knows which document it needs, rather than re-searching
- `compare_provisions(doc_ids, aspect, k_per_doc)` — parallel per-document search for
  side-by-side comparison questions ("how does X differ between issuers A and B")
- `extract_numeric_field(doc_id, field_description)` — makes its own LLM call
  (`claude-haiku-4-5`) against fetched text to pull out a specific number, for
  questions the other three tools can locate but not directly answer

Capped at `MAX_TOOL_TURNS = 8`. Every tool call and result is captured in a structured
trace (`AgentResult.trace`) that the dashboard renders directly — not just a final
answer, but the actual plan-call-observe sequence, viewable per query. The same
prompt-injection defense as `/query` applies: tool results are returned as JSON strings
and the agent's system prompt treats them as inert data.

The same MCP server used earlier for `search_corpus` alone (`agent/mcp_server.py`) now
exposes all 4 tools via `CorpusTools`, sharing the same retrieval/generation code paths
as the FastAPI endpoints — no duplicated logic between the MCP surface and the HTTP API.

## Tech stack

Python 3.12, `uv` for dependency management, `rank-bm25` for S0, local
`sentence-transformers` for S1 and S3 (bi-encoder + `ms-marco-MiniLM-L-6-v2`
cross-encoder, both CPU-friendly and free), FastAPI for serving, the Anthropic API
(Claude Sonnet 5 + Claude Haiku 4.5) and AWS Bedrock (`boto3`/`AnthropicBedrock`,
`us.anthropic.claude-sonnet-4-6`, verified live) behind a shared `ModelProvider`
abstraction, MCP for the tool-server surface, React + Vite + Plotly for the dashboard,
Docker for containerization. No vector database — chunks fit comfortably in memory as
a numpy matrix, and introducing one would have been complexity without a measured
benefit at this corpus size.
