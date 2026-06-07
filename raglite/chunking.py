from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    chunk_id: int
    text: str


def read_text_files(root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        files.append((path, path.read_text(encoding="utf-8")))
    return files


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    tokens = tokenize(text)
    if not tokens:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_id = 0
    step = chunk_size - overlap

    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        if chunks and len(window) <= overlap:
            break
        chunk = _untokenize(window).strip()
        if chunk:
            chunks.append(DocumentChunk(source=source, chunk_id=chunk_id, text=chunk))
            chunk_id += 1
        start += step

    return chunks


def load_chunks(
    docs_dir: Path,
    *,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path, text in read_text_files(docs_dir):
        chunks.extend(
            chunk_text(
                text,
                str(path),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    return chunks


def _untokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    return text
