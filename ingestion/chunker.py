"""Chunking strategies for Recall ingestion.

Three strategies, all producing fully-populated :class:`~models.Chunk` objects:

* ``FIXED``     — fixed token-count windows with overlap (tiktoken-based).
* ``RECURSIVE`` — LangChain ``RecursiveCharacterTextSplitter`` that respects
  paragraph / sentence boundaries.
* ``SEMANTIC``  — sentence embeddings, split where the cosine distance between
  adjacent sentences exceeds a threshold.

Use :func:`chunk` as the single dispatch entrypoint.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Callable, Protocol

from models import Chunk


class ChunkStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class SupportsEmbedding(Protocol):
    """Structural type for anything that can embed a list of texts.

    Recall's :class:`retrieval.embedder.Embedder` satisfies this, as does a raw
    sentence-transformers model (via ``encode``) and a plain callable.
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _section_of(doc: dict[str, Any]) -> str:
    """Derive a best-effort section label for a document.

    Args:
        doc: A loaded document dict (``{doc_id, text, source, metadata}``).

    Returns:
        The document title if available, otherwise an empty string.
    """
    meta = doc.get("metadata") or {}
    return str(meta.get("title") or meta.get("section") or "")


def _make_chunk(doc: dict[str, Any], text: str, index: int, strategy: ChunkStrategy) -> Chunk:
    """Construct a fully-populated :class:`~models.Chunk`.

    Args:
        doc: The source document dict.
        text: The chunk text.
        index: Ordinal index of this chunk within the document.
        strategy: The strategy that produced the chunk.

    Returns:
        A :class:`~models.Chunk` with all metadata fields populated.
    """
    return Chunk(
        chunk_id=f"{doc['doc_id']}-{strategy.value}-{index}",
        doc_id=doc["doc_id"],
        text=text.strip(),
        source=doc["source"],
        section=_section_of(doc),
        chunk_index=index,
        strategy=strategy.value,
    )


def _get_encoder():
    """Return a tiktoken encoder (cl100k_base) for token-based chunking.

    Returns:
        A tiktoken ``Encoding`` instance.
    """
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


# --------------------------------------------------------------------------- #
# Strategy: FIXED
# --------------------------------------------------------------------------- #
def chunk_fixed(doc: dict[str, Any], chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    """Split a document into fixed token-count windows with overlap.

    Args:
        doc: A loaded document dict (``{doc_id, text, source, metadata}``).
        chunk_size: Window size in tokens.
        overlap: Number of overlapping tokens between consecutive windows.

    Returns:
        A list of :class:`~models.Chunk` objects (strategy="fixed").

    Raises:
        ValueError: If ``overlap`` is not smaller than ``chunk_size``.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    enc = _get_encoder()
    tokens = enc.encode(doc["text"])
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    index = 0
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        text = enc.decode(window).strip()
        if text:
            chunks.append(_make_chunk(doc, text, index, ChunkStrategy.FIXED))
            index += 1
        if start + chunk_size >= len(tokens):
            break
    return chunks


# --------------------------------------------------------------------------- #
# Strategy: RECURSIVE
# --------------------------------------------------------------------------- #
def chunk_recursive(doc: dict[str, Any], chunk_size: int = 512, overlap: int = 50) -> list[Chunk]:
    """Split a document with LangChain's recursive character splitter.

    The splitter prefers to break on paragraph then sentence then word
    boundaries, keeping semantically-coherent units intact. Sizes are measured
    in tokens (via tiktoken) for consistency with :func:`chunk_fixed`.

    Args:
        doc: A loaded document dict (``{doc_id, text, source, metadata}``).
        chunk_size: Target chunk size in tokens.
        overlap: Token overlap between consecutive chunks.

    Returns:
        A list of :class:`~models.Chunk` objects (strategy="recursive").
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    pieces = splitter.split_text(doc["text"])
    return [
        _make_chunk(doc, piece, i, ChunkStrategy.RECURSIVE)
        for i, piece in enumerate(p for p in pieces if p.strip())
    ]


# --------------------------------------------------------------------------- #
# Strategy: SEMANTIC
# --------------------------------------------------------------------------- #
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences with a lightweight regex.

    Args:
        text: Input text.

    Returns:
        A list of non-empty, stripped sentences.
    """
    # First split on blank lines (hard paragraph breaks), then on sentence ends.
    sentences: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip())
    return sentences


def _embed_sentences(embedder: Any, sentences: list[str]) -> "list[list[float]]":
    """Embed sentences using whatever embedder interface is provided.

    Accepts (in order of preference): an object with ``embed_texts``, a
    sentence-transformers-style object with ``encode``, or a plain callable.

    Args:
        embedder: The embedding provider.
        sentences: Sentences to embed.

    Returns:
        A list of embedding vectors (one per sentence).

    Raises:
        TypeError: If ``embedder`` exposes none of the supported interfaces.
    """
    if hasattr(embedder, "embed_texts"):
        return embedder.embed_texts(sentences)
    if hasattr(embedder, "encode"):
        return [list(v) for v in embedder.encode(sentences)]
    if callable(embedder):
        return embedder(sentences)
    raise TypeError("embedder must have .embed_texts, .encode, or be callable")


def _cosine_distance(a: "list[float]", b: "list[float]") -> float:
    """Compute cosine distance (1 - cosine similarity) between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine distance in ``[0, 2]``.
    """
    import numpy as np

    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(1.0 - (va @ vb) / denom)


def chunk_semantic(
    doc: dict[str, Any],
    embedder: SupportsEmbedding | Callable[[list[str]], list[list[float]]],
    threshold: float = 0.3,
    max_sentences: int = 12,
) -> list[Chunk]:
    """Split a document at semantic boundaries between adjacent sentences.

    Each sentence is embedded; a new chunk begins whenever the cosine distance
    between consecutive sentences exceeds ``threshold`` (or a hard cap of
    ``max_sentences`` is reached to bound chunk size).

    Args:
        doc: A loaded document dict (``{doc_id, text, source, metadata}``).
        embedder: Embedding provider (``embed_texts`` / ``encode`` / callable).
        threshold: Cosine-distance threshold above which to start a new chunk.
        max_sentences: Hard cap on sentences per chunk.

    Returns:
        A list of :class:`~models.Chunk` objects (strategy="semantic").
    """
    sentences = _split_sentences(doc["text"])
    if not sentences:
        return []
    if len(sentences) == 1:
        return [_make_chunk(doc, sentences[0], 0, ChunkStrategy.SEMANTIC)]

    embeddings = _embed_sentences(embedder, sentences)

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        distance = _cosine_distance(embeddings[i - 1], embeddings[i])
        if distance > threshold or len(groups[-1]) >= max_sentences:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [
        _make_chunk(doc, " ".join(group), i, ChunkStrategy.SEMANTIC)
        for i, group in enumerate(groups)
    ]


# --------------------------------------------------------------------------- #
# Unified entrypoint
# --------------------------------------------------------------------------- #
def chunk(doc: dict[str, Any], strategy: ChunkStrategy, **kwargs: Any) -> list[Chunk]:
    """Chunk a document with the requested strategy.

    Args:
        doc: A loaded document dict (``{doc_id, text, source, metadata}``).
        strategy: Which :class:`ChunkStrategy` to apply.
        **kwargs: Strategy-specific options. ``SEMANTIC`` requires
            ``embedder=...``; ``FIXED``/``RECURSIVE`` accept ``chunk_size`` and
            ``overlap``.

    Returns:
        A list of :class:`~models.Chunk` objects.

    Raises:
        ValueError: If ``strategy`` is SEMANTIC but no ``embedder`` was given,
            or if the strategy is unknown.
    """
    if strategy == ChunkStrategy.FIXED:
        return chunk_fixed(doc, **kwargs)
    if strategy == ChunkStrategy.RECURSIVE:
        return chunk_recursive(doc, **kwargs)
    if strategy == ChunkStrategy.SEMANTIC:
        if "embedder" not in kwargs:
            raise ValueError("chunk_semantic requires an 'embedder' keyword argument")
        return chunk_semantic(doc, **kwargs)
    raise ValueError(f"Unknown chunking strategy: {strategy}")


if __name__ == "__main__":
    # Smoke test: exercise all three strategies. SEMANTIC uses a tiny
    # deterministic fake embedder so the test needs no heavy model downloads.
    import hashlib

    from rich.console import Console

    console = Console()

    sample = (
        "Photosynthesis converts light into chemical energy. "
        "It occurs in the chloroplasts of plant cells. "
        "The Roman Empire was founded in 27 BC. "
        "Augustus became its first emperor. "
        "Water boils at 100 degrees Celsius at sea level. "
        "Pressure changes alter the boiling point."
    )
    doc = {
        "doc_id": "smoke",
        "text": sample,
        "source": "smoke.txt",
        "metadata": {"title": "Mixed Facts"},
    }

    def fake_embedder(texts: list[str]) -> list[list[float]]:
        """Deterministic toy embedding: map each text to a small vector by hash."""
        vecs = []
        for t in texts:
            h = hashlib.md5(t.lower().encode()).digest()
            vecs.append([b / 255.0 for b in h[:8]])
        return vecs

    fixed = chunk_fixed(doc, chunk_size=20, overlap=5)
    recursive = chunk_recursive(doc, chunk_size=20, overlap=5)
    semantic = chunk_semantic(doc, fake_embedder, threshold=0.05)

    for name, result in [("fixed", fixed), ("recursive", recursive), ("semantic", semantic)]:
        console.print(f"[bold cyan]{name}[/]: {len(result)} chunk(s)")
        for c in result:
            assert c.strategy == name
            assert c.chunk_id and c.doc_id == "smoke" and c.text
            console.print(f"   [{c.chunk_index}] {c.text[:60]!r}")

    assert fixed and recursive and semantic, "every strategy should produce chunks"
    console.print("[bold green]chunker.py smoke test passed.[/]")
