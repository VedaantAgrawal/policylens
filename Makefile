.PHONY: setup ingest chunk embed test eval-s0 eval-s1 eval-s2 eval eval-generation eval-all ablation serve dashboard dashboard-deploy eval-latency-cost

# Fetches the corpus and builds every derived artifact from scratch.
# data/raw/ and data/processed/ are gitignored on purpose (see .gitignore) —
# this is how a clean clone gets them back.
setup: ingest chunk embed

ingest:
	uv run python -m policylens.ingest.fetch

chunk:
	uv run python -m policylens.ingest.chunk

embed:
	uv run python -m policylens.retrieval.embed

test:
	uv run pytest -q

eval-s0:
	uv run python -m policylens.eval.run_stage s0_bm25

eval-s1:
	uv run python -m policylens.eval.run_stage s1_dense

eval-s2:
	uv run python -m policylens.eval.run_stage s2_hybrid

# Retrieval-only ablation table (recall@k / MRR / nDCG@10). Free — no API
# key needed, everything runs locally (BM25 + local sentence-transformers).
eval: eval-s0 eval-s1 eval-s2
	uv run python -m policylens.eval.ablation

# Generation + groundedness + refusal accuracy. Requires ANTHROPIC_API_KEY
# in .env and makes real API calls (~$1-2 total at current pricing) —
# separate from `make eval` so reproducing the retrieval numbers never
# requires an API key or spends money.
eval-generation:
	uv run python -m policylens.eval.run_generation_stage s0_bm25
	uv run python -m policylens.eval.run_generation_stage s1_dense
	uv run python -m policylens.eval.run_generation_stage s2_hybrid
	uv run python -m policylens.eval.ablation

eval-all: eval eval-generation

# p50/p95 latency + cost-per-query for the recommended production stage
# (S2 hybrid). Requires ANTHROPIC_API_KEY, makes real API calls.
eval-latency-cost:
	uv run python -m policylens.eval.run_latency_cost s2_hybrid

ablation:
	uv run python -m policylens.eval.ablation

serve:
	uv run uvicorn policylens.serving.app:app --host 0.0.0.0 --port 8000

# Renders site/index.html from eval_results/*.json — no hand-typed numbers.
dashboard:
	uv run python scripts/build_dashboard.py

# Redeploys the static dashboard to Cloudflare Pages (policylens.vedaantagrawal.com).
# Requires `npx wrangler login` once per machine.
dashboard-deploy: dashboard
	npx wrangler deploy
