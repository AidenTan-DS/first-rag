from __future__ import annotations

import os
from textwrap import shorten

from .store import SearchResult


def answer_question(question: str, results: list[SearchResult], *, model: str) -> str:
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _answer_with_openai(question, results, model=model)
        except Exception as exc:
            fallback = _answer_without_llm(question, results)
            return f"{fallback}\n\n[OpenAI generation failed: {exc}]"

    return _answer_without_llm(question, results)


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        f"[{index}] Source: {result.source}#chunk-{result.chunk_id}\n{result.text}"
        for index, result in enumerate(results, start=1)
    )
    return (
        "Answer the user's question using only the context below. "
        "If the context is not enough, say you do not know. Cite sources like [1].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def _answer_with_openai(question: str, results: list[SearchResult], *, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        temperature=0,
        input=[
            {
                "role": "system",
                "content": "You are a factual RAG assistant. Ground every answer in the provided context.",
            },
            {"role": "user", "content": build_prompt(question, results)},
        ],
    )
    return response.output_text


def _answer_without_llm(question: str, results: list[SearchResult]) -> str:
    if not results:
        return "I could not find relevant context in the index."

    lines = [
        "No OPENAI_API_KEY found, so here are the most relevant retrieved chunks.",
        f"Question: {question}",
        "",
    ]
    for index, result in enumerate(results, start=1):
        preview = shorten(result.text.replace("\n", " "), width=500, placeholder="...")
        lines.append(
            f"[{index}] score={result.score:.3f} source={result.source}#chunk-{result.chunk_id}\n{preview}\n"
        )
    return "\n".join(lines).strip()
