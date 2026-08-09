# PolicyLens dashboard

React + Vite + Plotly frontend: a chat view (agent or direct-RAG mode) and an
eval view charting the retrieval ablation. Talks to the FastAPI backend in
`../src/policylens/serving/app.py` — see `src/api.js` for the endpoints used
(`/query`, `/agent/query`, `/eval`). No numbers are hand-typed; the eval view
fetches live from the running API, which serves the same committed
`eval_results/*.json` files `make eval` regenerates.

```sh
npm install
npm run dev      # local dev server, hits http://localhost:8000 by default
npm run build     # production build to dist/, hits the deployed API
                    #   (see .env.production)
```

Deployed to Cloudflare Pages via `make dashboard-deploy` from the repo root.
