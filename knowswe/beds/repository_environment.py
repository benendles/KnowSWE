from dataclasses import dataclass
from pathlib import Path

from knowswe.beds.adaptive_actions import (
    ActionType,
    InvestigationAction,
)
from knowswe.repository.dependency_graph import RepositoryGraph
from knowswe.retrieval.bm25 import RepositoryRetriever


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


@dataclass
class RepositoryObservation:
    content: str
    discovered_actions: list[InvestigationAction]


class RepositoryInvestigationEnvironment:
    """
    Execute file-level repository investigation actions.

    BEDS-2 prototype:
    - BM25 initial retrieval
    - File inspection
    - Import-dependency discovery
    - Estimated evidence-token costs

    No benchmark ground truth is accessed.
    """

    def __init__(self, repository_path: str):
        self.root = Path(repository_path).resolve()

        if not self.root.is_dir():
            raise ValueError("Repository does not exist")

        self.retriever = RepositoryRetriever(str(self.root))
        self.graph = RepositoryGraph(str(self.root))

        self.retriever.index()
        self.graph.build()

    def _read_source(self, filename: str) -> str:
        path = (self.root / filename).resolve()

        if not path.is_relative_to(self.root):
            raise ValueError("Path escapes repository")

        if not path.is_file():
            raise FileNotFoundError(filename)

        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

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

        for seed in seeds:
            filename = seed["file"]
            content = self._read_source(filename)

            relevance = (
                max(seed["score"], 0.0) / maximum
                if maximum > 0
                else 0.0
            )

            actions.append(
                InvestigationAction(
                    action_type=ActionType.INSPECT,
                    target=filename,
                    estimated_cost=estimate_tokens(content),
                    relevance=relevance,
                    diagnostic_value=relevance,
                )
            )

        return actions

    def execute(
        self,
        action: InvestigationAction,
    ) -> RepositoryObservation:

        if action.action_type not in {
            ActionType.INSPECT,
            ActionType.FOLLOW_DEPENDENCY,
            ActionType.INSPECT_TEST,
        }:
            raise ValueError(
                f"Unsupported action: {action.action_type}"
            )

        content = self._read_source(action.target)

        dependencies = self.graph.dependencies(
            action.target,
            depth=1,
        )

        discovered = []

        for filename in dependencies:
            dependency_content = self._read_source(filename)

            discovered.append(
                InvestigationAction(
                    action_type=ActionType.FOLLOW_DEPENDENCY,
                    target=filename,
                    estimated_cost=estimate_tokens(
                        dependency_content
                    ),
                    relevance=max(
                        action.relevance * 0.8,
                        0.0,
                    ),
                    diagnostic_value=max(
                        action.diagnostic_value * 0.8,
                        0.0,
                    ),
                )
            )

        return RepositoryObservation(
            content=content,
            discovered_actions=discovered,
        )
