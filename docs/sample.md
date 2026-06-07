# RAG Notes

Retrieval augmented generation, usually shortened to RAG, is a pattern for answering questions using external documents.

The indexing stage happens before the user asks a question. It loads documents, splits them into chunks, embeds every chunk, and stores those vectors with the original text and source metadata.

The query stage happens for each user question. It embeds the question with the same embedding method, searches for the most similar chunks, builds a prompt with those chunks, and asks a language model to answer using only the provided context.

Chunk size matters. Chunks that are too small lose context, while chunks that are too large can hide the relevant fact among too many unrelated words. A practical starting point is a few hundred tokens with some overlap.

RAG is useful when the answer depends on private, recent, or changing information. It is usually not the right tool for pure creative writing or general knowledge that the model already knows.
