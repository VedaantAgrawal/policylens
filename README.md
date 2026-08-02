# PolicyLens

Agentic RAG over public insurance and regulatory documents (SEC EDGAR life-insurer
10-Ks, NAIC model laws, state Department of Insurance bulletins) — built around a
measured retrieval ablation, not a chat demo.

Every number below comes from a committed file in [`eval_results/`](eval_results/),
regenerated from scratch by `make eval` (retrieval, free) and `make eval-generation`
(generation + judge, requires an Anthropic API key, ~$1-2 in API spend). Nothing here
is hand-typed.

**Live:** [interactive dashboard](https://policylens.vedaantagrawal.com) (static, Cloudflare Pages) ·
[API](https://policylens-api.vedaantagrawal.com/health) (FastAPI behind a Cloudflare Tunnel to a local Docker container)

## Results

### Retrieval ablation (recall@k, MRR, nDCG@10 — 95% bootstrap CI, n=30 answerable questions)

| Stage | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| **S0** BM25 (baseline) | 0.600 [0.433, 0.767] | 0.767 [0.600, 0.900] | 0.465 [0.319, 0.611] | 0.535 [0.398, 0.668] |
| **S1** dense (all-MiniLM-L6-v2) | 0.517 [0.333, 0.700] | 0.533 [0.367, 0.700] | 0.426 [0.263, 0.590] | 0.447 [0.285, 0.608] |
| **S2** hybrid (RRF, S0+S1) | 0.633 [0.467, 0.800] | 0.767 [0.600, 0.900] | 0.495 [0.345, 0.641] | 0.556 [0.412, 0.689] |

**Dense retrieval alone is worse than the BM25 baseline on this corpus.** This golden
set skews toward precise-terminology questions ("how does the NAIC model law define
X") where exact lexical match beats a small, non-domain-tuned embedding model. Hybrid
fusion recovers the gap and comes out ahead of both individual stages on every
metric — but a paired bootstrap on the S0→S2 delta (10k resamples) shows **none of
these deltas are statistically significant at n=30** (every 95% CI crosses zero). The
point estimates favor hybrid; the golden set needs to grow before that's a confident
claim rather than a plausible one.

### Generation quality (citation precision, groundedness, refusal accuracy — 95% bootstrap CI)

| Retriever used for generation | citation precision | groundedness | refusal accuracy | false refusal rate |
|---|---|---|---|---|
| S0 BM25 | 1.000 [1.000, 1.000] (n=26) | 1.000 [1.000, 1.000] (n=24) | 0.917 [0.750, 1.000] (n=12) | 0.200 [0.067, 0.367] (n=30) |
| S1 dense | 1.000 [1.000, 1.000] (n=24) | 1.000 [1.000, 1.000] (n=21) | 0.917 [0.750, 1.000] (n=12) | 0.300 [0.133, 0.467] (n=30) |
| S2 hybrid | 1.000 [1.000, 1.000] (n=25) | 0.905 [0.762, 1.000] (n=21) | 0.917 [0.750, 1.000] (n=12) | 0.300 [0.133, 0.467] (n=30) |

Generation runs on `claude-sonnet-5` with a structured JSON output (answer + inline
`[chunk_id]` citations + a self-reported `answerable` flag); groundedness is judged by
`claude-haiku-4-5` against a written claim-by-claim rubric; citation precision is a
deterministic check (cited chunk actually in the retrieved set), not an LLM judgment.

Refusal accuracy is identical across all three stages (11/12), and groundedness/false-
refusal differences between them (e.g. S2's 0.905 vs S0/S1's 1.000 groundedness) sit
well inside each other's confidence intervals — read this as "generation quality is
roughly stage-invariant here," not "hybrid retrieval hurts generation." Each question
was judged once, not resampled, so day-to-day model variance is part of that CI width.

**False refusals are a retrieval problem, not a generation problem.** I traced every
false refusal under S2 by hand: of the 9 answerable questions the model refused, **0
had the gold chunk in the retrieved top-5** — the generator was never handed the right
context and correctly declined to guess. Refusal calibration itself looks solid
(91.7% accuracy on genuinely unanswerable questions, with zero cases of refusing
despite having the answer available).

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
  retrieval/{bm25,dense,hybrid}.py  ◄───────────────────┘
        │                                    eval/runner.py
        ▼                                    (recall@k, MRR, nDCG@10,
  generation/generate.py                      bootstrap CI)
  (cited synthesis, refusal path)                    │
        │                                            ▼
        ▼                                    eval_results/*.json (committed)
  eval/judge.py (groundedness)                        │
        │                                             ▼
        ▼                                    eval/ablation.py → this README
  serving/app.py (FastAPI: /query, /eval, /health)
```

- **Corpus** (109 docs, ~6,500 chunks): 6 SEC 10-Ks from publicly traded life
  insurers (MetLife, Prudential, Lincoln National, Brighthouse, Corebridge, Unum —
  mutual companies like MassMutual don't file 10-Ks), 45 NAIC model laws
  (life/annuity/reinsurance/solvency-relevant, confirmed free at content.naic.org),
  54 state DOI/DFS bulletins (NY, CA, TX — Massachusetts was dropped after mass.gov
  returned 403 to automated requests; that's a WAF, not a robots.txt disallow, and
  bypassing it isn't "official API/bulk data" access).
- **Chunking** is section-aware: PDF page boundaries and HTML `<h1-6>` tags plus a
  line-shape heuristic (`Item 7.`, `Section 3.`, `ARTICLE I`, all-caps headers) drive
  chunk boundaries, so `section_heading` and `page` metadata on each chunk are
  accurate to what's actually in it — that's what inline citations point back to.
- **Golden set** (42 questions: 30 answerable + 12 unanswerable) is hand-written
  against real sampled chunk text, not LLM-generated — every `relevant_chunk_id` is
  verified by script to exist in the corpus before being committed. The unanswerable
  subset covers distinct refusal-trap categories: entities not in corpus (mutual
  companies), jurisdictions not in corpus (MA/FL/WA), a NAIC model law deliberately
  excluded from the curated list (Long-Term Care #640, to test against false-positive
  retrieval), stale/future-dated questions, out-of-domain questions, and unverifiable
  precision claims.
- **Model-provider abstraction**: `providers/base.py` defines a one-method
  `ModelProvider` protocol; `AnthropicProvider` (live) and `BedrockProvider` (real
  interface, untested — no AWS credentials in this environment) both implement it, so
  generation/judge code never imports an SDK directly.
- **Prompt-injection defense**: retrieved chunk text is wrapped in `<source>` tags
  with an explicit system-prompt instruction to treat it as inert data, never as
  instructions — a chunk containing "ignore prior instructions and..." is
  attacker-controlled content the model must not act on.

## Reproduce it

```sh
git clone https://github.com/VedaantAgrawal/policylens.git
cd policylens
uv sync

make setup             # fetch corpus, chunk, embed (~2 min, needs internet)
make test               # 46 unit tests, including the eval metrics themselves
make eval               # regenerate the retrieval ablation table — free, no API key
make eval-generation     # regenerate generation/groundedness/refusal — needs
                          #   ANTHROPIC_API_KEY in .env, costs ~$1-2 in API spend
```

`make eval` alone reproduces every number in the retrieval ablation table above from
a clean clone with zero API cost — BM25 and the dense embedding model
(`all-MiniLM-L6-v2`) both run locally. `make eval-generation` is separate and
explicit about spending money, so the free path stays free.

To run the API locally:

```sh
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
make serve               # FastAPI on :8000
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "How many days after a premium due date must a policyholder request a paid-up nonforfeiture benefit under the NAIC Standard Nonforfeiture Law?"}'
```

## Known limitations

- **n=30 answerable questions is small.** Retrieval-stage deltas are not statistically
  significant at this sample size (see above) — treat the ablation as a methodology
  demonstration, not a settled result. Growing the golden set is the highest-value
  next step for making the comparisons load-bearing.
- **Dense embeddings use a small, general-purpose model** (`all-MiniLM-L6-v2`, 384-dim,
  not domain-tuned), chosen to run locally at zero cost. A larger or
  insurance/legal-tuned embedding model would likely close or reverse the S0/S1 gap.
- **Chunking has a known artifact**: NAIC PDF page footers repeat the document title
  (e.g. "Insurance Data Security Model Law MO-668-2 (c) 2017...") on every page, which
  occasionally inflates lexical similarity for chunks adjacent to the correct one on
  title-mentioning queries. Documented in the S0 commit, not silently patched over.
- **S3 (cross-encoder rerank) was cut for time**, not because it's uninteresting —
  the retrieval interface (`retrieval/base.py`) is designed so adding it is a new
  file, not a refactor.
- **Agent/MCP layer is a stub, not a working agent** (see below) — cut deliberately to
  protect the eval harness under a compressed timeline, per an explicit priority call
  made when this project was scoped.

## Agent/MCP layer (design, not implemented)

The full architecture called for an MCP server exposing `search_corpus`,
`fetch_section`, `compare_provisions`, and `extract_numeric_field` tools, with an
agent loop that plans, calls tools, and synthesizes with a structured trace. Under a
3-day timeline, this was cut in favor of the eval harness and ablation table — the
`/query` endpoint above is single-shot retrieval + generation, not a multi-step agent.
`search_corpus` is a thin wrapper over `HybridRetriever.search`, and is the only piece
actually stubbed out, to make the cut concrete rather than purely aspirational.

## Tech stack

Python 3.12, `uv` for dependency management, `rank-bm25` for S0, local
`sentence-transformers` for S1, FastAPI for serving, the Anthropic API for
generation/judging (Claude Sonnet 5 + Claude Haiku 4.5), Docker for containerization.
No vector database — 6,500 chunks fit comfortably in memory as a numpy matrix, and
introducing one would have been complexity without a measured benefit at this corpus
size.
