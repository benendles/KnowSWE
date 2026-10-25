"""Gemma-assisted, budgeted repository investigation."""

from dataclasses import dataclass, field
from time import perf_counter

from knowswe.beds.adaptive_controller import AdaptiveBEDSController
from knowswe.beds.chunk_environment import ChunkInvestigationEnvironment
from knowswe.beds.hypotheses import HypothesisTracker


@dataclass(frozen=True)
class GemmaInvestigationStep:
    step: int
    location: str
    support: int
    evidence_line: int | None
    evidence_text: str | None
    confidence_before: float
    confidence_after: float
    inspection_cost: int
    inference_seconds: float


@dataclass
class GemmaInvestigationResult:
    state: object
    hypotheses: HypothesisTracker
    assessments: list[GemmaInvestigationStep] = field(
        default_factory=list
    )


def investigate_with_gemma(
    repository_path: str,
    bug_report: str,
    evaluator,
    budget: int = 2000,
    top_k: int = 5,
    max_steps: int = 10,
    max_chunk_tokens: int = 300,
    relevance_mode: str = "chunk",
) -> GemmaInvestigationResult:
    """Run BEDS inspections with validated Gemma assessments.

    The budget currently covers inspected source tokens only.
    Model inference tokens are not yet included.
    """

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

    assessments = []

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

            started = perf_counter()

            assessment = evaluator.assess(
                bug_report=bug_report,
                location=selected.target,
                content=observation.content,
                start_line=observation.chunk.start_line,
            )

            inference_seconds = perf_counter() - started

            if assessment.location != selected.target:
                raise ValueError("Assessment location mismatch")

            if (
                type(assessment.support) is not int
                or assessment.support not in (-1, 0, 1)
            ):
                raise ValueError("Invalid assessment support")

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

        tracker.update(
            location=selected.target,
            evidence_id=selected.target,
            support=assessment.support,
        )

        assessments.append(
            GemmaInvestigationStep(
                step=step_number,
                location=selected.target,
                support=assessment.support,
                evidence_line=assessment.evidence_line,
                evidence_text=assessment.evidence_text,
                confidence_before=confidence_before,
                confidence_after=hypothesis.confidence,
                inspection_cost=actual_cost,
                inference_seconds=inference_seconds,
            )
        )

    if (
        controller.state.stop_reason is None
        and max_steps > 0
        and len(assessments) == max_steps
    ):
        controller.state.stop_reason = "max_steps_reached"

    return GemmaInvestigationResult(
        state=controller.state,
        hypotheses=tracker,
        assessments=assessments,
    )
