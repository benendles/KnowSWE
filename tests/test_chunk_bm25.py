import unittest

from knowswe.chunking.method_chunker import MethodChunk
from knowswe.retrieval.chunk_bm25 import ChunkBM25Retriever


def make_chunk(symbol, content, start):
    return MethodChunk(
        filename="example.py",
        start_line=start,
        end_line=start + 2,
        symbol=symbol,
        parent_symbol=None,
        content=content,
        estimated_tokens=max(1, len(content) // 4),
    )


class TestChunkBM25(unittest.TestCase):

    def test_empty_index(self):
        retriever = ChunkBM25Retriever([])
        self.assertEqual(retriever.search("bug"), [])

    def test_returns_all_chunks(self):
        chunks = [
            make_chunk("alpha", "invoice payment calculation", 1),
            make_chunk("beta", "network socket connection", 5),
            make_chunk("gamma", "database transaction rollback", 9),
        ]

        retriever = ChunkBM25Retriever(chunks)

        results = retriever.search("invoice payment")

        self.assertEqual(len(results), 3)

    def test_relevant_chunk_ranks_first(self):
        chunks = [
            make_chunk("alpha", "invoice payment calculation", 1),
            make_chunk("beta", "network socket connection", 5),
            make_chunk("gamma", "database transaction rollback", 9),
        ]

        retriever = ChunkBM25Retriever(chunks)

        results = retriever.search("invoice payment")

        self.assertEqual(results[0][0].symbol, "alpha")


if __name__ == "__main__":
    unittest.main()
