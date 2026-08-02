FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

# Corpus, chunks, and embeddings are gitignored (regenerable, see .gitignore) —
# build them into the image so the container doesn't need external network
# access to SEC/NAIC/state sites or Hugging Face at every startup.
RUN uv run python -m policylens.ingest.fetch \
 && uv run python -m policylens.ingest.chunk \
 && uv run python -m policylens.retrieval.embed

EXPOSE 8000

# ANTHROPIC_API_KEY is passed at `docker run` time (--env-file .env), never
# baked into the image.
CMD ["uv", "run", "uvicorn", "policylens.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
