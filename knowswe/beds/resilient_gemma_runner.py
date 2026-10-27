"""Fault-tolerant Gemma investigation with explicit failure records."""

from dataclasses import dataclass, field

from knowswe.beds.adaptive_controller import AdaptiveBEDSController
from knowswe.beds.chunk_environment import ChunkInvestigationEnvironment
from knowswe.beds.hypotheses import HypothesisTracker
from knowswe.beds.adaptive_reranking import rerank_pending_actions


@dataclass(frozen=True)
class ResilientInvestigationStep:
    step: int
    location: str
    success: bool
    support: int | None
    evidence_line: int | None
    confidence_before: float
    confidence_after: float
    inspection_cost: int
    inference_attempts: int
    inference_seconds: float
    errors: tuple[str, ...] = ()


@dataclass
class ResilientInvestigationResult:
    state: object
    hypotheses: HypothesisTracker
    steps: list[ResilientInvestigationStep] = field(
        default_factory=list
    )


def investigate_resiliently(
    repository_path: str,
    bug_report: str,
    evaluator,
    budget: int = 2000,
    top_k: int = 5,
    max_steps: int = 10,
    max_chunk_tokens: int = 300,
    relevance_mode: str = "chunk",
    enable_propagation: bool = False,
) -> ResilientInvestigationResult:

    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")

    environment = ChunkInvestigationEnvironment(
        repository_path,
        max_chunk_tokens=max_chunk_tokens,
        relevance_mode=relevance_mode,
    )

    controller = AdaptiveBEDSController(budget=budget)

    actions = environment.initial_actions(
        bug_report,
        top_k=top_k,
    )

    controller.add_actions(actions)

    tracker = HypothesisTracker()

    for action in actions:
        tracker.add(
            action.target,
            confidence=min(
                1.0,
                max(0.0, action.relevance),
            ),
        )

    steps = []

    for step_number in range(1, max_steps + 1):
        selected = controller.select_next()

        if selected is None:
            controller.state.stop_reason = "no_affordable_actions"
            break

        controller.begin_action(selected)

        try:
            observation = environment.execute(selected)
            actual_cost = observation.chunk.estimated_tokens

            if actual_cost > controller.state.remaining_budget:
                controller.state.stop_reason = (
                    "observation_exceeds_remaining_budget"
                )
                controller.cancel_action()
                break

            result = evaluator.assess(
                bug_report=bug_report,
                location=selected.target,
                content=observation.content,
                start_line=observation.chunk.start_line,
            )

            # Source inspection is charged even if inference failed.
            controller.observe(
                selected,
                observation.content,
                actual_cost=actual_cost,
            )

        except Exception:
            controller.cancel_action()
            raise

        hypothesis = tracker.hypotheses[selected.target]
        confidence_before = hypothesis.confidence

        assessment = result.assessment

        if assessment is not None:
            if assessment.location != selected.target:
                raise ValueError("Assessment location mismatch")

            if (
                type(assessment.support) is not int
                or assessment.support not in (-1, 0, 1)
            ):
                raise ValueError("Invalid assessment support")

            tracker.update(
                location=selected.target,
                evidence_id=selected.target,
                support=assessment.support,
            )

            if enable_propagation and assessment.support != 0:
                rerank_pending_actions(
                    controller=controller,
                    tracker=tracker,
                    chunks=environment.chunks,
                    source_location=selected.target,
                    support=assessment.support,
                )

        steps.append(
            ResilientInvestigationStep(
                step=step_number,
                location=selected.target,
                success=result.success,
                support=(
                    assessment.support
                    if assessment is not None
                    else None
                ),
                evidence_line=(
                    assessment.evidence_line
                    if assessment is not None
                    else None
                ),
                confidence_before=confidence_before,
                confidence_after=hypothesis.confidence,
                inspection_cost=actual_cost,
                inference_attempts=len(result.attempts),
                inference_seconds=result.total_seconds,
                errors=tuple(
                    attempt.error
                    for attempt in result.attempts
                    if attempt.error is not None
                ),
            )
        )

    if (
        controller.state.stop_reason is None
        and max_steps > 0
        and len(steps) == max_steps
    ):
        controller.state.stop_reason = "max_steps_reached"

    return ResilientInvestigationResult(
        state=controller.state,
        hypotheses=tracker,
        steps=steps,
    )
