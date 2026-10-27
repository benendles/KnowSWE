"""Structural evidence propagation between repository code chunks."""

from dataclasses import dataclass

from knowswe.chunking.method_chunker import MethodChunk


@dataclass(frozen=True)
class PropagationResult:
    source: str
    target: str
    relationship: str
    weight: float
    support: int
    confidence_delta: float


def structural_relationship(
    source: MethodChunk,
    target: MethodChunk,
) -> tuple[str, float]:
    """Return the strongest structural relationship between two chunks."""

    if source.chunk_id == target.chunk_id:
        return "same_chunk", 0.0

    if source.filename != target.filename:
        return "different_file", 0.0

    if (
        source.symbol == target.symbol
        and source.symbol
        and source.symbol != "<module>"
    ):
        return "same_method", 1.0

    if (
        source.parent_symbol
        and target.parent_symbol
        and source.parent_symbol == target.parent_symbol
    ):
        return "same_class", 0.5

    return "same_file", 0.15


def propagate_evidence(
    source: MethodChunk,
    candidates: list[MethodChunk],
    support: int,
    learning_rate: float = 0.2,
) -> list[PropagationResult]:
    """Compute heuristic confidence changes for related candidate chunks."""

    if type(support) is not int or support not in (-1, 0, 1):
        raise ValueError("support must be -1, 0, or 1")

    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError("learning_rate must be between 0 and 1")

    results = []

    for target in candidates:
        relationship, weight = structural_relationship(source, target)

        if weight == 0.0 or support == 0:
            continue

        results.append(
            PropagationResult(
                source=source.chunk_id,
                target=target.chunk_id,
                relationship=relationship,
                weight=weight,
                support=support,
                confidence_delta=learning_rate * support * weight,
            )
        )

    return results
