"""Document loading for Recall ingestion.

Loads raw source documents of several file types into a uniform shape so the
chunker downstream never has to care where the text came from.

Supported file types: ``.txt``, ``.md``, ``.pdf``, ``.json``, ``.html``/``.htm``.

Each loaded document is a plain ``dict`` with the keys::

    {"doc_id": str, "text": str, "source": str, "metadata": dict}

(The ``Chunk`` Pydantic model is produced later, by the chunker.)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

SUPPORTED_SUFFIXES: set[str] = {".txt", ".md", ".pdf", ".json", ".html", ".htm"}

# Common JSON keys that hold the primary document text.
_JSON_TEXT_KEYS = ("text", "content", "body", "article", "page_content")


def _make_doc_id(source: Path, salt: str = "") -> str:
    """Build a short, stable document id from a source path.

    Args:
        source: Path of the source document.
        salt: Extra disambiguator (e.g. an index when one file yields many docs).

    Returns:
        A 16-character hex document id.
    """
    digest = hashlib.sha1(f"{source.as_posix()}::{salt}".encode()).hexdigest()
    return digest[:16]


def _load_txt(path: Path) -> list[dict[str, Any]]:
    """Load a plain-text or markdown file.

    Args:
        path: Path to the ``.txt`` / ``.md`` file.

    Returns:
        A single-element list with the document dict.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return [
        {
            "doc_id": _make_doc_id(path),
            "text": text,
            "source": path.as_posix(),
            "metadata": {"filetype": path.suffix.lstrip(".")},
        }
    ]


def _load_pdf(path: Path) -> list[dict[str, Any]]:
    """Load a PDF file, concatenating page text.

    Args:
        path: Path to the ``.pdf`` file.

    Returns:
        A single-element list with the document dict (page count in metadata).
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n\n".join(pages).strip()
    return [
        {
            "doc_id": _make_doc_id(path),
            "text": text,
            "source": path.as_posix(),
            "metadata": {"filetype": "pdf", "num_pages": len(reader.pages)},
        }
    ]


def _load_html(path: Path) -> list[dict[str, Any]]:
    """Load an HTML file, extracting visible text via BeautifulSoup.

    Args:
        path: Path to the ``.html`` / ``.htm`` file.

    Returns:
        A single-element list with the document dict (title in metadata).
    """
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return [
        {
            "doc_id": _make_doc_id(path),
            "text": text,
            "source": path.as_posix(),
            "metadata": {"filetype": "html", "title": title},
        }
    ]


def _extract_json_text(obj: Any) -> str:
    """Best-effort extraction of text from an arbitrary JSON object.

    Args:
        obj: A decoded JSON value (dict, list, str, ...).

    Returns:
        The most text-like representation we can find.
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in _JSON_TEXT_KEYS:
            if isinstance(obj.get(key), str):
                return obj[key]
        # Fall back to a readable dump of string values.
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> list[dict[str, Any]]:
    """Load a JSON file as one document, or many if it contains a list.

    Args:
        path: Path to the ``.json`` file.

    Returns:
        One document dict per top-level record (or one for a single object).
    """
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    records = data if isinstance(data, list) else [data]
    docs: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        meta = {"filetype": "json"}
        if isinstance(record, dict):
            meta.update({k: v for k, v in record.items() if isinstance(v, (str, int, float, bool))})
        docs.append(
            {
                "doc_id": _make_doc_id(path, salt=str(i)),
                "text": _extract_json_text(record),
                "source": f"{path.as_posix()}#{i}" if len(records) > 1 else path.as_posix(),
                "metadata": meta,
            }
        )
    return docs


_LOADERS = {
    ".txt": _load_txt,
    ".md": _load_txt,
    ".pdf": _load_pdf,
    ".json": _load_json,
    ".html": _load_html,
    ".htm": _load_html,
}


def load_document(path: str | Path) -> list[dict[str, Any]]:
    """Load a single document file into one or more document dicts.

    Args:
        path: Path to a supported document file.

    Returns:
        A list of document dicts, each ``{doc_id, text, source, metadata}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file type is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(f"Unsupported file type '{suffix}' for {path}. Supported: {sorted(SUPPORTED_SUFFIXES)}")
    return _LOADERS[suffix](path)


def load_documents(path: str | Path) -> list[dict[str, Any]]:
    """Load every supported document under a path (file or directory).

    Args:
        path: A single file or a directory to scan recursively.

    Returns:
        A flat list of document dicts ``{doc_id, text, source, metadata}``,
        skipping empty documents and unsupported file types.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such path: {path}")

    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)

    docs: list[dict[str, Any]] = []
    for file in files:
        try:
            loaded = load_document(file)
        except Exception as exc:  # noqa: BLE001 - report and continue ingesting the rest
            console.print(f"[yellow]Skipping {file}: {exc}[/]")
            continue
        for doc in loaded:
            if doc["text"].strip():
                docs.append(doc)
            else:
                console.print(f"[yellow]Skipping empty document: {doc['source']}[/]")

    console.print(f"[green]Loaded {len(docs)} document(s) from {path}[/]")
    return docs


if __name__ == "__main__":
    # Smoke test: write a couple of temp files and load them.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "a.txt").write_text("Hello world. This is a plain text document.", encoding="utf-8")
        (tmp_path / "b.json").write_text(
            json.dumps([{"text": "First record."}, {"content": "Second record."}]), encoding="utf-8"
        )
        (tmp_path / "c.html").write_text(
            "<html><head><title>T</title></head><body><p>Body text here.</p></body></html>", encoding="utf-8"
        )
        loaded = load_documents(tmp_path)
        for d in loaded:
            console.print(f"  [cyan]{d['doc_id']}[/] {d['source']} -> {d['text'][:40]!r}")
        assert len(loaded) == 4, f"expected 4 docs, got {len(loaded)}"
        console.print("[bold green]loader.py smoke test passed.[/]")
