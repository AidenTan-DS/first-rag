from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from textwrap import shorten

from .chunking import load_chunks
from .embeddings import HashingEmbedder
from .llm import answer_question, build_prompt
from .store import VectorStore


DEFAULT_INDEX = Path(".rag_index/index.json")


def main() -> int:
    parser = ArgumentParser(description="A tiny local RAG pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Chunk and index documents.")
    ingest.add_argument("docs_dir", type=Path, help="Directory containing .md or .txt files.")
    ingest.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ingest.add_argument("--chunk-size", type=int, default=180)
    ingest.add_argument("--overlap", type=int, default=40)
    ingest.add_argument("--dimensions", type=int, default=384)

    chunks = subparsers.add_parser("chunks", help="Show how documents are split into chunks.")
    chunks.add_argument("docs_dir", type=Path, help="Directory containing .md or .txt files.")
    chunks.add_argument("--chunk-size", type=int, default=180)
    chunks.add_argument("--overlap", type=int, default=40)
    chunks.add_argument("--limit", type=int, default=5)

    search = subparsers.add_parser("search", help="Show retrieved chunks without generation.")
    search.add_argument("question")
    search.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--dimensions", type=int, default=384)

    prompt = subparsers.add_parser("prompt", help="Show the final prompt sent to the LLM.")
    prompt.add_argument("question")
    prompt.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    prompt.add_argument("--top-k", type=int, default=5)
    prompt.add_argument("--dimensions", type=int, default=384)

    ask = subparsers.add_parser("ask", help="Ask a question against the indexed documents.")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--dimensions", type=int, default=384)
    ask.add_argument("--model", default="gpt-4.1-mini")

    args = parser.parse_args()

    if args.command == "ingest":
        return _ingest(args.docs_dir, args.index, args.chunk_size, args.overlap, args.dimensions)
    if args.command == "chunks":
        return _chunks(args.docs_dir, args.chunk_size, args.overlap, args.limit)
    if args.command == "search":
        return _search(args.question, args.index, args.top_k, args.dimensions)
    if args.command == "prompt":
        return _prompt(args.question, args.index, args.top_k, args.dimensions)
    if args.command == "ask":
        return _ask(args.question, args.index, args.top_k, args.dimensions, args.model)

    parser.error(f"Unknown command: {args.command}")
    return 2


def _ingest(
    docs_dir: Path,
    index_path: Path,
    chunk_size: int,
    overlap: int,
    dimensions: int,
) -> int:
    if not docs_dir.exists():
        raise SystemExit(f"Docs directory does not exist: {docs_dir}")

    chunks = load_chunks(docs_dir, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise SystemExit(f"No .md or .txt documents found in {docs_dir}")

    store = VectorStore.from_chunks(chunks, HashingEmbedder(dimensions=dimensions))
    store.save(index_path)
    print(f"Indexed {len(chunks)} chunks into {index_path}")
    return 0


def _chunks(docs_dir: Path, chunk_size: int, overlap: int, limit: int) -> int:
    if not docs_dir.exists():
        raise SystemExit(f"Docs directory does not exist: {docs_dir}")

    chunks = load_chunks(docs_dir, chunk_size=chunk_size, overlap=overlap)
    print(f"Loaded {len(chunks)} chunks")
    for chunk in chunks[:limit]:
        preview = shorten(chunk.text.replace("\n", " "), width=500, placeholder="...")
        print(f"\n{chunk.source}#chunk-{chunk.chunk_id}\n{preview}")
    return 0


def _search(question: str, index_path: Path, top_k: int, dimensions: int) -> int:
    results = _retrieve(question, index_path, top_k, dimensions)
    for index, result in enumerate(results, start=1):
        preview = shorten(result.text.replace("\n", " "), width=500, placeholder="...")
        print(
            f"[{index}] score={result.score:.3f} "
            f"source={result.source}#chunk-{result.chunk_id}\n{preview}\n"
        )
    return 0


def _prompt(question: str, index_path: Path, top_k: int, dimensions: int) -> int:
    results = _retrieve(question, index_path, top_k, dimensions)
    print(build_prompt(question, results))
    return 0


def _ask(question: str, index_path: Path, top_k: int, dimensions: int, model: str) -> int:
    results = _retrieve(question, index_path, top_k, dimensions)
    print(answer_question(question, results, model=model))
    return 0


def _retrieve(question: str, index_path: Path, top_k: int, dimensions: int):
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}. Run `python3 -m raglite ingest docs` first.")

    store = VectorStore.load(index_path, HashingEmbedder(dimensions=dimensions))
    return store.search(question, top_k=top_k)
