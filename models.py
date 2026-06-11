"""Shared Pydantic v2 data models for the Recall RAG system.

Every piece of data that crosses a module boundary in Recall is one of these
models — never a bare ``dict`` — so that retrieval, generation, and eval all
agree on a single, validated schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single chunk of source text plus its provenance metadata.

    Attributes:
        chunk_id: Stable unique identifier for this chunk.
        doc_id: Identifier of the source document the chunk came from.
        text: The chunk's textual content.
        source: Human-readable source path / URL of the document.
        section: Section or heading the chunk belongs to (best effort).
        chunk_index: Ordinal position of the chunk within its document.
        strategy: Chunking strategy used ("fixed" | "recursive" | "semantic").
        embedding: Dense embedding vector, populated once embedded.
    """

    chunk_id: str
    doc_id: str
    text: str
    source: str
    section: str
    chunk_index: int
    strategy: str
    embedding: list[float] | None = None


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever, with its score and provenance.

    Attributes:
        chunk: The underlying :class:`Chunk`.
        score: Retriever score (cosine similarity, BM25, or RRF fusion score).
        retriever: Which retriever produced it ("dense" | "sparse" | "hybrid").
        reranker_score: Cross-encoder rerank score, if reranking was applied.
    """

    chunk: Chunk
    score: float
    retriever: str
    reranker_score: float | None = None


class Answer(BaseModel):
    """A grounded answer plus the full retrieval/generation trace.

    Attributes:
        query: The user query that was answered.
        answer_text: The generated answer text (with inline ``[n]`` citations).
        cited_chunk_ids: Chunk ids the answer actually cited.
        retrieved_chunks: Chunks returned by retrieval (pre-rerank).
        reranked_chunks: Chunks after reranking (the ones shown to the LLM).
        latency_ms: End-to-end pipeline latency in milliseconds.
        run_config: The config dict that produced this answer.
    """

    query: str
    answer_text: str
    cited_chunk_ids: list[str]
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    latency_ms: float
    run_config: dict


class EvalResult(BaseModel):
    """The result of evaluating one pipeline configuration over the golden set.

    Attributes:
        run_id: Unique identifier for this experiment run.
        config: The pipeline configuration evaluated.
        timestamp: ISO-8601 timestamp of when the run finished.
        scores: Aggregate metric scores (context_recall, context_precision,
            faithfulness, answer_relevancy).
        per_question: Per-question records (trace + per-question scores).
    """

    run_id: str
    config: dict
    timestamp: str
    scores: dict
    per_question: list[dict] = Field(default_factory=list)


if __name__ == "__main__":
    # Smoke test: construct each model and round-trip through JSON.
    from rich import print as rprint

    chunk = Chunk(
        chunk_id="c0",
        doc_id="d0",
        text="The mitochondria is the powerhouse of the cell.",
        source="data/raw/bio.txt",
        section="Cell Biology",
        chunk_index=0,
        strategy="recursive",
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.91, retriever="hybrid", reranker_score=8.4)
    answer = Answer(
        query="What is the powerhouse of the cell?",
        answer_text="The mitochondria [1].",
        cited_chunk_ids=["c0"],
        retrieved_chunks=[retrieved],
        reranked_chunks=[retrieved],
        latency_ms=123.4,
        run_config={"chunking_strategy": "recursive"},
    )
    result = EvalResult(
        run_id="smoke",
        config={"chunking_strategy": "recursive"},
        timestamp="2026-06-10T00:00:00Z",
        scores={"faithfulness": 1.0},
        per_question=[{"question_id": "q0", "faithfulness": 1.0}],
    )
    for name, model in [
        ("Chunk", chunk),
        ("RetrievedChunk", retrieved),
        ("Answer", answer),
        ("EvalResult", result),
    ]:
        rprint(f"[bold green]{name}[/] round-trips OK:", model.model_validate_json(model.model_dump_json()) is not None)
    rprint("[bold green]models.py smoke test passed.[/]")
