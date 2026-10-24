from pathlib import Path

from knowswe.beds.controller import Candidate, BEDSController
from knowswe.retrieval.bm25 import RepositoryRetriever
from knowswe.repository.dependency_graph import RepositoryGraph


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class BudgetedRepositoryInspector:
    def __init__(self, repository_path: str, budget: int = 1000):
        self.root = Path(repository_path).resolve()
        self.retriever = RepositoryRetriever(repository_path)
        self.graph = RepositoryGraph(repository_path)
        self.budget = budget

    def investigate(self, bug_report: str, top_k: int = 1, depth: int = 2):
        controller = BEDSController(budget=self.budget)

        self.retriever.index()
        self.graph.build()

        seeds = self.retriever.search(bug_report, top_k=top_k)
        candidates = {}

        for seed in seeds:
            filename = seed["file"]

            distances = {
                filename: 0,
                **self.graph.dependency_distances(filename, depth=depth),
            }

            for candidate_file, distance in distances.items():
                path = (self.root / candidate_file).resolve()

                if not path.is_relative_to(self.root) or not path.is_file():
                    continue

                content = path.read_text(
                    encoding="utf-8",
                    errors="replace"
                )

                candidate = Candidate(
                    filename=candidate_file,
                    relevance=seed["score"] if distance == 0 else 0.0,
                    distance=distance,
                    cost=estimate_tokens(content),
                )

                previous = candidates.get(candidate_file)

                if previous is None or distance < previous.distance:
                    candidates[candidate_file] = candidate

        remaining = list(candidates.values())
        selected_files = []

        while remaining:
            selected = controller.select(remaining)

            if selected is None:
                break

            remaining.remove(selected)

            selected_files.append({
                "file": selected.filename,
                "estimated_tokens": selected.cost,
                "distance": selected.distance,
                "score": controller.score(selected),
            })

        return {
            "bug_report": bug_report,
            "budget": controller.budget,
            "remaining_budget": controller.remaining,
            "selected_files": selected_files,
        }
