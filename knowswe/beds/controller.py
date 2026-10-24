from dataclasses import dataclass


@dataclass
class Candidate:
    filename: str
    relevance: float
    distance: int
    cost: int


class BEDSController:
    def __init__(
        self,
        budget: int = 1000,
        relevance_weight: float = 0.5,
        distance_weight: float = 0.3,
        cost_weight: float = 0.2,
    ):
        if budget <= 0:
            raise ValueError("Budget must be positive")

        self.budget = budget
        self.remaining = budget

        self.relevance_weight = relevance_weight
        self.distance_weight = distance_weight
        self.cost_weight = cost_weight

    def score(self, candidate: Candidate):
        relevance = max(candidate.relevance, 0.0)
        proximity = 1.0 / (1.0 + candidate.distance)
        cost_ratio = candidate.cost / self.budget

        return (
            self.relevance_weight * relevance
            + self.distance_weight * proximity
            - self.cost_weight * cost_ratio
        )

    def select(self, candidates: list[Candidate]):
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
                candidate.filename,
            ),
        )

        self.remaining -= selected.cost

        return selected
