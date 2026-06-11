# Recall

> A production-grade retrieval-augmented generation system with a metric-driven eval harness — hybrid search, cross-encoder reranking, grounded citations, and a measurable improvement loop.

## Architecture

```
                ┌─────────────┐
   raw docs ──▶ │  Ingestion  │  load → chunk (fixed | recursive | semantic) → embed
                └──────┬──────┘
                       ▼
              ┌──────────────────┐        ┌──────────────┐
              │   Vector store   │◀──────▶│   BM25 idx   │
              │   (Qdrant)       │        └──────────────┘
              └────────┬─────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Retrieval                    │
        │  dense ─┐                     │
        │  sparse ─┼─▶ RRF hybrid ─▶ rerank (cross-encoder)
        └──────────────────────────────┘
                       ▼
              ┌──────────────────┐
              │   Generation     │  grounded answer + inline [n] citations
              │   (Claude)       │
              └────────┬─────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Eval harness (RAGAS)         │  context recall / precision /
        │  + deepeval CI quality gates  │  faithfulness / answer relevancy
        └──────────────────────────────┘
                       ▼
            FastAPI trace API · Streamlit dashboard
```

[placeholder for diagram]

## Eval Results

| Run | Chunking  | Retrieval | Reranker | Context Recall | Context Precision | Faithfulness | Answer Relevancy |
|-----|-----------|-----------|----------|----------------|-------------------|--------------|------------------|
| 1   | fixed     | dense     | no       | —              | —                 | —            | —                |
| 2   | recursive | hybrid    | no       | —              | —                 | —            | —                |
| 3   | recursive | hybrid    | yes      | —              | —                 | —            | —                |
| 4   | semantic  | hybrid    | yes      | —              | —                 | —            | —                |

## Stack

- **Python 3.11+**, **uv** for environment + dependency management
- **Qdrant** (local Docker) — dense vector store
- **Claude `claude-sonnet-4-20250514`** (Anthropic SDK) — generation + LLM judge
- **sentence-transformers** — `BAAI/bge-large-en-v1.5` embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker
- **rank_bm25** — sparse retrieval
- **RAGAS** — retrieval/answer eval metrics
- **deepeval** — pytest-style CI quality gates
- **FastAPI** — trace viewer API
- **Streamlit** — eval dashboard UI
- **rich** — CLI output

## Setup

```bash
# 1. Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment + install dependencies (uv manages Python 3.11)
uv sync

# 3. Configure secrets
cp .env.example .env   # then fill in ANTHROPIC_API_KEY

# 4. Start Qdrant
docker compose up -d qdrant
```

## Running Experiments

```bash
# Ingest a corpus with a chunking strategy (Phase 1)
uv run python scripts/ingest.py --strategy recursive --corpus data/raw/

# Run a full eval experiment (Phase 5)
uv run python scripts/run_experiment.py --chunking recursive --retrieval hybrid --rerank true --top_k 20

# Compare two runs
uv run python scripts/compare_runs.py --a <run_id_1> --b <run_id_2>

# CI quality gates
uv run pytest tests/

# Observability
uv run uvicorn api.main:app --reload          # trace API
uv run streamlit run dashboard/app.py          # dashboard
```

> Note: modules are also runnable directly for smoke tests, e.g. `uv run python -m ingestion.chunker`.

## What I Learned About RAG Failure Modes

_(To be filled in as experiments run — e.g. chunk-boundary recall loss, lexical vs.
semantic retrieval trade-offs, reranker precision gains, citation hallucination, etc.)_
