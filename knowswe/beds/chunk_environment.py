"""Chunk-level repository environment for adaptive BEDS."""

from dataclasses import dataclass
from pathlib import Path

from knowswe.beds.adaptive_actions import (
    ActionType,
    InvestigationAction,
)
from knowswe.chunking.method_chunker import (
    MethodChunk,
    chunk_python_methods,
)
from knowswe.retrieval.bm25 import RepositoryRetriever
from knowswe.retrieval.chunk_bm25 import ChunkBM25Retriever


def structural_content_score(chunk: MethodChunk) -> float:
    """Heuristic estimate of how much executable code a chunk contains."""

    meaningful_lines = [
        line.strip()
        for line in chunk.content.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
    ]

    if not meaningful_lines:
        return 0.0

    if chunk.estimated_tokens < 10:
        return 0.0

    code_density = len(meaningful_lines) / max(
        len(chunk.content.splitlines()),
        1,
    )

    size_factor = min(
        chunk.estimated_tokens / 100.0,
        1.0,
    )

    return code_density * size_factor


@dataclass
class ChunkObservation:
    content: str
    chunk: MethodChunk
    discovered_actions: list[InvestigationAction]


class ChunkInvestigationEnvironment:

    def __init__(
        self,
        repository_path: str,
        max_chunk_tokens: int = 300,
        relevance_mode: str = 'file',
    ):
        self.root = Path(repository_path).resolve()

        if not self.root.is_dir():
            raise ValueError("Repository directory not found")

        if max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")

        if relevance_mode not in {"file", "chunk"}:
            raise ValueError("Invalid relevance mode")

        self.relevance_mode = relevance_mode
        self.max_chunk_tokens = max_chunk_tokens

        self.retriever = RepositoryRetriever(str(self.root))
        self.retriever.index()

        self.chunks: dict[str, MethodChunk] = {}

    def initial_actions(
        self,
        bug_report: str,
        top_k: int = 5,
    ) -> list[InvestigationAction]:

        seeds = self.retriever.search(
            bug_report,
            top_k=top_k,
        )

        if not seeds:
            return []

        maximum = max(
            max(seed["score"], 0.0)
            for seed in seeds
        )

        actions = []
        candidates = []

        for seed in seeds:
            filename = seed["file"]

            if not filename.endswith(".py"):
                continue

            path = (self.root / filename).resolve()

            if not path.is_relative_to(self.root):
                continue

            if not path.is_file():
                continue

            chunks = chunk_python_methods(
                path,
                self.root,
                max_tokens=self.max_chunk_tokens,
            )

            file_relevance = (
                max(seed["score"], 0.0) / maximum
                if maximum > 0
                else 0.0
            )

            for chunk in chunks:
                candidates.append((chunk, file_relevance))

        chunk_scores = {}

        if self.relevance_mode == "chunk" and candidates:
            retriever = ChunkBM25Retriever(
                [chunk for chunk, _ in candidates]
            )

            ranked = retriever.search(bug_report)

            max_chunk_score = max(
                (max(score, 0.0) for _, score in ranked),
                default=0.0,
            )

            chunk_scores = {
                chunk.chunk_id: (
                    max(score, 0.0) / max_chunk_score
                    if max_chunk_score > 0
                    else 0.0
                )
                for chunk, score in ranked
            }

        for chunk, file_relevance in candidates:
            content_score = structural_content_score(chunk)

            if content_score <= 0:
                continue

            self.chunks[chunk.chunk_id] = chunk

            relevance = (
                chunk_scores[chunk.chunk_id]
                if self.relevance_mode == "chunk"
                else file_relevance
            )

            actions.append(
                InvestigationAction(
                    action_type=ActionType.INSPECT,
                    target=chunk.chunk_id,
                    estimated_cost=chunk.estimated_tokens,
                    relevance=relevance,
                    diagnostic_value=content_score,
                )
            )

        return actions

    def execute(
        self,
        action: InvestigationAction,
    ) -> ChunkObservation:

        if action.action_type != ActionType.INSPECT:
            raise ValueError("Unsupported chunk action type")

        chunk = self.chunks.get(action.target)

        if chunk is None:
            raise ValueError("Unknown chunk identifier")

        return ChunkObservation(
            content=chunk.content,
            chunk=chunk,
            discovered_actions=[],
        )
