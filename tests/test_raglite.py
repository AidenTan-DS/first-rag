from pathlib import Path
import tempfile
import unittest

from raglite.chunking import chunk_text
from raglite.embeddings import HashingEmbedder
from raglite.store import VectorStore


class ChunkingTest(unittest.TestCase):
    def test_chunks_keep_overlap(self) -> None:
        text = " ".join(f"word{i}" for i in range(12))

        chunks = chunk_text(text, "test.txt", chunk_size=6, overlap=2)

        self.assertEqual(len(chunks), 3)
        self.assertIn("word4 word5", chunks[0].text)
        self.assertTrue(chunks[1].text.startswith("word4 word5"))

    def test_tiny_overlap_tail_is_skipped(self) -> None:
        text = " ".join(f"word{i}" for i in range(10))

        chunks = chunk_text(text, "test.txt", chunk_size=6, overlap=4)

        self.assertEqual(chunks[-1].text, "word4 word5 word6 word7 word8 word9")


class VectorStoreTest(unittest.TestCase):
    def test_search_returns_related_chunk_first(self) -> None:
        chunks = chunk_text(
            "Indexing splits documents and embeds chunks. Cooking pasta needs boiling water.",
            "notes.txt",
            chunk_size=7,
            overlap=0,
        )
        store = VectorStore.from_chunks(chunks, HashingEmbedder(dimensions=64))

        results = store.search("How do documents get embedded?", top_k=1)

        self.assertIn("embeds chunks", results[0].text)

    def test_store_round_trips_to_json(self) -> None:
        chunks = chunk_text("RAG retrieves context before generation.", "notes.txt")
        store = VectorStore.from_chunks(chunks, HashingEmbedder(dimensions=32))

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "index.json"
            store.save(index_path)
            loaded = VectorStore.load(index_path, HashingEmbedder(dimensions=32))

        self.assertEqual(loaded.chunks[0].text, store.chunks[0].text)
        self.assertEqual(len(loaded.chunks[0].embedding), 32)


if __name__ == "__main__":
    unittest.main()
