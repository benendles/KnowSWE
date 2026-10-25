"""BM25 retrieval over method-aware source chunks."""

from rank_bm25 import BM25Okapi

from knowswe.chunking.method_chunker import MethodChunk
from knowswe.retrieval.bm25 import tokenize


class ChunkBM25Retriever:

    def __init__(self, chunks: list[MethodChunk]):
        self.chunks = chunks

        self.documents = [
            tokenize(
                f"{chunk.symbol} "
                f"{chunk.filename} "
                f"{chunk.content}"
            )
            for chunk in chunks
        ]

        self.model = (
            BM25Okapi(self.documents)
            if self.documents
            else None
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[MethodChunk, float]]:

        if self.model is None:
            return []

        scores = self.model.get_scores(tokenize(query))

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda item: (
                -float(item[1]),
                item[0].chunk_id,
            ),
        )

        if top_k is not None:
            ranked = ranked[:top_k]

        return [
            (chunk, float(score))
            for chunk, score in ranked
        ]
