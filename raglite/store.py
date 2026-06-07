from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from .chunking import DocumentChunk
from .embeddings import HashingEmbedder, cosine_similarity


@dataclass(frozen=True)
class IndexedChunk:
    source: str
    chunk_id: int
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    source: str
    chunk_id: int
    text: str
    score: float


class VectorStore:
    def __init__(self, chunks: list[IndexedChunk], embedder: HashingEmbedder) -> None:
        self.chunks = chunks
        self.embedder = embedder

    @classmethod
    def from_chunks(cls, chunks: list[DocumentChunk], embedder: HashingEmbedder) -> "VectorStore":
        indexed = [
            IndexedChunk(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedder.embed(chunk.text),
            )
            for chunk in chunks
        ]
        return cls(indexed, embedder)

    @classmethod
    def load(cls, path: Path, embedder: HashingEmbedder) -> "VectorStore":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["dimensions"] != embedder.dimensions:
            raise ValueError(
                f"Index dimensions are {data['dimensions']}, but embedder has {embedder.dimensions}"
            )
        chunks = [IndexedChunk(**item) for item in data["chunks"]]
        return cls(chunks, embedder)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimensions": self.embedder.dimensions,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        query_embedding = self.embedder.embed(query)
        scored = [
            SearchResult(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=cosine_similarity(query_embedding, chunk.embedding),
            )
            for chunk in self.chunks
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]
