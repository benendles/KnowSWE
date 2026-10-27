"""Evidence-guided reranking of pending BEDS investigation actions."""

from dataclasses import dataclass, replace

from knowswe.beds.evidence_propagation import propagate_evidence


@dataclass(frozen=True)
class RerankingRecord:
    source: str
    target: str
    relationship: str
    support: int
    confidence_before: float
    confidence_after: float
    score_before: float
    score_after: float


def rerank_pending_actions(
    controller,
    tracker,
    chunks,
    source_location: str,
    support: int,
    learning_rate: float = 0.2,
) -> list[RerankingRecord]:
    """Propagate evidence and update pending action relevance.

    The controller's action score is recalculated automatically
    when select_next() is called again.

    Confidence values are heuristic scores, not calibrated probabilities.
    """

    if type(support) is not int or support not in (-1, 0, 1):
        raise ValueError("support must be -1, 0, or 1")

    if source_location not in chunks:
        raise KeyError(f"Unknown source chunk: {source_location}")

    if source_location not in tracker.hypotheses:
        raise KeyError(f"Unknown source hypothesis: {source_location}")

    if support == 0:
        return []

    source_chunk = chunks[source_location]

    pending = list(controller.state.pending_actions.items())

    candidate_chunks = [
        chunks[action.target]
        for _, action in pending
        if action.target in chunks
        and action.target in tracker.hypotheses
    ]

    propagation = propagate_evidence(
        source=source_chunk,
        candidates=candidate_chunks,
        support=support,
        learning_rate=learning_rate,
    )

    records = []

    for change in propagation:
        hypothesis = tracker.hypotheses[change.target]

        action_key = next(
            (
                key
                for key, action in pending
                if action.target == change.target
            ),
            None,
        )

        if action_key is None:
            continue

        action = controller.state.pending_actions[action_key]

        before = hypothesis.confidence
        score_before = controller.score(action)

        after = max(
            0.0,
            min(1.0, before + change.confidence_delta),
        )

        hypothesis.confidence = after

        evidence_id = (
            f"propagated:{source_location}:{change.relationship}"
        )

        if support > 0:
            hypothesis.supporting_evidence.append(evidence_id)
        else:
            hypothesis.contradicting_evidence.append(evidence_id)

        controller.state.pending_actions[action_key] = replace(
            action,
            relevance=after,
        )

        score_after = controller.score(
            controller.state.pending_actions[action_key]
        )

        records.append(
            RerankingRecord(
                source=source_location,
                target=change.target,
                relationship=change.relationship,
                support=support,
                confidence_before=before,
                confidence_after=after,
                score_before=score_before,
                score_after=score_after,
            )
        )

    return records
