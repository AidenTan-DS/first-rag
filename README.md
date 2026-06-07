# Minimal RAG

A small, locally runnable RAG project built from first principles.

The pipeline is:

```text
documents -> chunk -> embed -> store
query -> embed -> retrieve -> prompt -> answer
```

This version is intentionally simple:

- Retrieval works without installing any external dependencies.
- When `OPENAI_API_KEY` is available, the project uses OpenAI to generate the final answer.
- Without an API key, it displays the retrieved context and sources so you can inspect the RAG pipeline.

## Quick Start

```bash
python3 -m raglite ingest docs
python3 -m raglite ask "What happens during RAG indexing?"
```

The index is stored in `.rag_index/index.json`.

## Learn It Module By Module

The project is divided into modules that follow a real RAG pipeline. Work through them in this order:

### 1. Chunking

First, inspect how documents are split into chunks:

```bash
python3 -m raglite chunks docs --chunk-size 80 --overlap 20
```

Module: `raglite/chunking.py`

Look for two things:

- Each chunk should contain enough context.
- The overlap should preserve some repeated context between adjacent chunks.

### 2. Embedding + Store

Convert the chunks into vectors and write them to the local index:

```bash
python3 -m raglite ingest docs --chunk-size 80 --overlap 20
```

Modules:

- `raglite/embeddings.py`
- `raglite/store.py`

This learning version uses a local `HashingEmbedder` to make the pipeline easy to inspect. A production project can replace it with an OpenAI embedding model.

### 3. Retrieval

Inspect retrieval results without asking an LLM to generate an answer:

```bash
python3 -m raglite search "What happens during indexing?"
```

Module: `raglite/store.py`

If retrieval returns the wrong chunks, the final answer will usually be wrong too.

### 4. Prompt

Inspect the final prompt that would be sent to the LLM:

```bash
python3 -m raglite prompt "What happens during indexing?"
```

Module: `raglite/llm.py`

A RAG prompt normally contains system instructions, retrieved context, and the user's question.

### 5. Answer

Finally, generate an answer:

```bash
python3 -m raglite ask "What happens during indexing?"
```

Without `OPENAI_API_KEY`, the command displays the retrieved chunks. With a key, it asks an LLM to answer using those chunks.

## Add Your Own Docs

Place `.txt` or `.md` files in `docs/`, then rebuild the index:

```bash
python3 -m raglite ingest docs
python3 -m raglite ask "Your question"
```

## Use OpenAI For Generation

```bash
export OPENAI_API_KEY="your-api-key"
python3 -m raglite ask "How does RAG answer after retrieving context?"
```

The default model is `gpt-4.1-mini`. You can override it:

```bash
python3 -m raglite ask "Your question" --model gpt-4.1
```

## Project Layout

```text
raglite/
  chunking.py    Document chunking
  embeddings.py  Local hashing embeddings
  store.py       JSON vector index and cosine retrieval
  llm.py         OpenAI generation and local fallback
  cli.py         CLI entry point and inspection commands
docs/
  sample.md      Example knowledge base
```

## Next Upgrades

After running the minimal version, common upgrades include:

1. Replace `HashingEmbedder` with `text-embedding-3-small`.
2. Replace the JSON store with Chroma, FAISS, LanceDB, or pgvector.
3. Add PDF and web page loaders.
4. Add evaluations that check retrieved chunks, answer quality, and source citations.
