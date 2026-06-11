"""Ingestion CLI: raw documents -> chunks -> vector store + BM25 index.

Pipeline:

1. Load every supported document under ``--corpus``.
2. Chunk them with the requested ``--strategy``.
3. Persist chunks as JSON under ``data/chunks/{strategy}/``.
4. Build a BM25 sparse index and pickle it alongside the chunks.
5. Embed all chunks and upsert them into the Qdrant collection
   ``recall_{strategy}`` (skippable with ``--skip-embed``).

Usage::

    python scripts/ingest.py --strategy recursive --corpus data/raw/
    python scripts/ingest.py --strategy fixed --corpus data/raw/ --skip-embed
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import uuid
from pathlib import Path

# Make the project root importable when this file is run directly by path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ingestion.chunker import ChunkStrategy, chunk
from ingestion.loader import load_documents
from models import Chunk

console = Console()

DATA_ROOT = _ROOT / "data"
CHUNKS_ROOT = DATA_ROOT / "chunks"

# A stable namespace so chunk_id -> Qdrant point UUID is deterministic.
_QDRANT_NAMESPACE = uuid.UUID("a3f1c2d4-0000-4000-8000-000000000000")
_TOKEN_RE = re.compile(r"\b\w+\b")


def _progress() -> Progress:
    """Create a configured rich progress bar.

    Returns:
        A :class:`rich.progress.Progress` instance with sensible columns.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 (lowercased word tokens).

    Args:
        text: Input text.

    Returns:
        A list of lowercase word tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def save_chunks(chunks: list[Chunk], strategy: ChunkStrategy) -> Path:
    """Persist chunks to ``data/chunks/{strategy}/chunks.json``.

    Args:
        chunks: The chunks to save.
        strategy: The chunking strategy (determines the subfolder).

    Returns:
        The path the chunks were written to.
    """
    out_dir = CHUNKS_ROOT / strategy.value
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.json"
    payload = [c.model_dump(exclude={"embedding"}) for c in chunks]
    import json

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Saved {len(chunks)} chunks ->[/] {out_path}")
    return out_path


def build_bm25_index(chunks: list[Chunk], strategy: ChunkStrategy) -> Path:
    """Build and pickle a BM25 index for the chunks.

    The pickle stores the fitted ``BM25Okapi`` model plus the chunk payloads in
    the same order, so the sparse retriever can map ranks back to chunks.

    Args:
        chunks: The chunks to index.
        strategy: The chunking strategy (determines the subfolder).

    Returns:
        The path the BM25 index was written to.
    """
    from rank_bm25 import BM25Okapi

    out_dir = CHUNKS_ROOT / strategy.value
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bm25_index.pkl"

    tokenized_corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    payload = {
        "bm25": bm25,
        "chunks": [c.model_dump(exclude={"embedding"}) for c in chunks],
    }
    with out_path.open("wb") as fh:
        pickle.dump(payload, fh)
    console.print(f"[green]Built BM25 index ({len(chunks)} docs) ->[/] {out_path}")
    return out_path


def embed_and_upsert(chunks: list[Chunk], strategy: ChunkStrategy, embedding_model: str, qdrant_url: str) -> None:
    """Embed chunks and upsert them into Qdrant.

    Args:
        chunks: The chunks to embed and store.
        strategy: The chunking strategy (collection is ``recall_{strategy}``).
        embedding_model: sentence-transformers model id.
        qdrant_url: Base URL of the Qdrant instance.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    from sentence_transformers import SentenceTransformer

    collection = f"recall_{strategy.value}"
    console.print(f"[cyan]Loading embedding model[/] {embedding_model} ...")
    model = SentenceTransformer(embedding_model)
    dim = model.get_sentence_embedding_dimension()

    texts = [c.text for c in chunks]
    with _progress() as progress:
        task = progress.add_task("[cyan]Embedding chunks", total=len(texts))
        vectors: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            vectors.extend(v.tolist() for v in emb)
            progress.update(task, advance=len(batch))

    client = QdrantClient(url=qdrant_url)
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
    )

    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid5(_QDRANT_NAMESPACE, c.chunk_id)),
            vector=vec,
            payload=c.model_dump(exclude={"embedding"}),
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection, points=points)
    console.print(f"[green]Upserted {len(points)} vectors ->[/] Qdrant collection '{collection}'")


def ingest(corpus: Path, strategy: ChunkStrategy, skip_embed: bool, embedding_model: str, qdrant_url: str) -> list[Chunk]:
    """Run the full ingestion pipeline for one corpus + strategy.

    Args:
        corpus: Path to the corpus (file or directory).
        strategy: Chunking strategy to apply.
        skip_embed: If True, skip embedding + Qdrant upsert (Phase 1 only).
        embedding_model: sentence-transformers model id (used unless skipping).
        qdrant_url: Qdrant base URL (used unless skipping).

    Returns:
        The list of produced chunks.
    """
    console.rule(f"[bold]Ingest · strategy={strategy.value} · corpus={corpus}")
    documents = load_documents(corpus)
    if not documents:
        console.print("[yellow]No documents loaded — nothing to ingest.[/]")
        return []

    embedder = None
    if strategy == ChunkStrategy.SEMANTIC:
        # Semantic chunking needs an embedder up front (independent of Qdrant).
        from sentence_transformers import SentenceTransformer

        console.print(f"[cyan]Loading embedding model for semantic chunking[/] {embedding_model} ...")
        embedder = SentenceTransformer(embedding_model)

    all_chunks: list[Chunk] = []
    with _progress() as progress:
        task = progress.add_task("[cyan]Chunking documents", total=len(documents))
        for doc in documents:
            kwargs = {"embedder": embedder} if strategy == ChunkStrategy.SEMANTIC else {}
            all_chunks.extend(chunk(doc, strategy, **kwargs))
            progress.update(task, advance=1)

    console.print(f"[green]Produced {len(all_chunks)} chunks from {len(documents)} documents.[/]")

    save_chunks(all_chunks, strategy)
    build_bm25_index(all_chunks, strategy)

    if skip_embed:
        console.print("[yellow]--skip-embed set: skipping embedding + Qdrant upsert.[/]")
    elif all_chunks:
        embed_and_upsert(all_chunks, strategy, embedding_model, qdrant_url)

    console.print("[bold green]Ingestion complete.[/]")
    return all_chunks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional explicit argument list (defaults to ``sys.argv``).

    Returns:
        The parsed arguments namespace.
    """
    import os

    parser = argparse.ArgumentParser(description="Ingest raw docs into chunks + vector store.")
    parser.add_argument(
        "--strategy",
        type=ChunkStrategy,
        choices=list(ChunkStrategy),
        default=ChunkStrategy(os.getenv("CHUNK_STRATEGY", "recursive")),
        help="Chunking strategy.",
    )
    parser.add_argument("--corpus", type=Path, default=DATA_ROOT / "raw", help="Corpus path (file or directory).")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding + Qdrant upsert (Phase 1 only).")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
        help="sentence-transformers model id.",
    )
    parser.add_argument(
        "--qdrant-url", type=str, default=os.getenv("QDRANT_URL", "http://localhost:6333"), help="Qdrant base URL."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint.

    Args:
        argv: Optional explicit argument list (defaults to ``sys.argv``).
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001 - dotenv is optional
        pass

    args = parse_args(argv)
    ingest(
        corpus=args.corpus,
        strategy=args.strategy,
        skip_embed=args.skip_embed,
        embedding_model=args.embedding_model,
        qdrant_url=args.qdrant_url,
    )


if __name__ == "__main__":
    main()
