from dataclasses import dataclass


@dataclass
class ChunkCandidate:
    index: int
    filename: str
    relevance: float
    cost: int


class ChunkBEDSController:
    """
    Static budget-aware chunk selection.

    This is an experimental selection heuristic, not yet
    the full adaptive BEDS agent.
    """

    def __init__(
        self,
        budget=2000,
        relevance_weight=0.70,
        diversity_weight=0.20,
        cost_weight=0.10,
    ):
        if budget <= 0:
            raise ValueError("Budget must be positive")

        self.budget = budget
        self.remaining = budget

        self.relevance_weight = relevance_weight
        self.diversity_weight = diversity_weight
        self.cost_weight = cost_weight

        self.selected_files = set()

    def score(self, candidate):
        relevance = max(candidate.relevance, 0.0)

        diversity = (
            1.0
            if candidate.filename not in self.selected_files
            else 0.0
        )

        cost_ratio = candidate.cost / self.budget

        return (
            self.relevance_weight * relevance
            + self.diversity_weight * diversity
            - self.cost_weight * cost_ratio
        )

    def select(self, candidates):
        available = [
            candidate
            for candidate in candidates
            if 0 < candidate.cost <= self.remaining
        ]

        if not available:
            return None

        selected = max(
            available,
            key=lambda candidate: (
                self.score(candidate),
                -candidate.cost,
                -candidate.index,
            ),
        )

        self.remaining -= selected.cost
        self.selected_files.add(selected.filename)

        return selected
