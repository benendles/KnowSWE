"""Evidence-driven fault-location hypothesis tracking."""

from dataclasses import dataclass, field


@dataclass
class FaultHypothesis:
    location: str
    confidence: float
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    inspected: bool = False


class HypothesisTracker:
    """Maintain and update candidate fault-location hypotheses."""

    def __init__(self):
        self.hypotheses: dict[str, FaultHypothesis] = {}

    def add(self, location: str, confidence: float = 0.5):
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

        if location not in self.hypotheses:
            self.hypotheses[location] = FaultHypothesis(
                location=location,
                confidence=confidence,
            )

    def update(
        self,
        location: str,
        evidence_id: str,
        support: float,
    ):
        """
        Update confidence using signed evidence.

        support:
            -1.0 = strongly contradicting
             0.0 = neutral
             1.0 = strongly supporting
        """

        if location not in self.hypotheses:
            raise KeyError(f"Unknown hypothesis: {location}")

        if not -1.0 <= support <= 1.0:
            raise ValueError("Support must be between -1 and 1")

        hypothesis = self.hypotheses[location]

        if support > 0:
            hypothesis.supporting_evidence.append(evidence_id)
        elif support < 0:
            hypothesis.contradicting_evidence.append(evidence_id)

        hypothesis.confidence = max(
            0.0,
            min(
                1.0,
                hypothesis.confidence + 0.2 * support,
            ),
        )

        hypothesis.inspected = True

    def ranked(self) -> list[FaultHypothesis]:
        return sorted(
            self.hypotheses.values(),
            key=lambda item: (-item.confidence, item.location),
        )
