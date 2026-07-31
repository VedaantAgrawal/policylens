# PolicyLens

Agentic RAG over public insurance and regulatory documents (SEC EDGAR life-insurer
filings, NAIC model laws, state Department of Insurance bulletins), built around a
measured retrieval ablation rather than a chat demo.

**Status: under active development.** Results table and reproduction instructions
land here once the first eval run is committed — see [eval_results/](eval_results/)
for raw run output as it becomes available.

## Why this project

Most portfolio RAG projects show a demo. This one shows a measurement: each
retrieval upgrade (BM25 baseline → dense embeddings → hybrid fusion → cross-encoder
rerank) is scored against the same golden question set, with bootstrap confidence
intervals on the deltas, so the comparisons are statistically honest rather than
vibes-based.
