.PHONY: ingest chunk test eval-s0 eval

ingest:
	uv run python -m policylens.ingest.fetch

chunk:
	uv run python -m policylens.ingest.chunk

test:
	uv run pytest -q

eval-s0:
	uv run python -m policylens.eval.run_stage s0_bm25

# Regenerates every number in the ablation table from scratch. Grows as
# S1/S2/S3 stages are added (Day 2) — `make eval` will run all of them.
eval: eval-s0
