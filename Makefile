.PHONY: ingest chunk embed test eval-s0 eval-s1 eval

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

eval-s1: embed
	uv run python -m policylens.eval.run_stage s1_dense

# Regenerates every number in the ablation table from scratch. Grows as
# S2/S3 stages are added — `make eval` will run all of them.
eval: eval-s0 eval-s1
