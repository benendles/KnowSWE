"""Hypothesis-guided chunk investigation.

The evidence evaluator is injected so that the adaptive loop
can be tested independently of a language model.
"""

from dataclasses import dataclass
from typing import Callable

from knowswe.beds.adaptive_actions import InvestigationAction
from knowswe.beds.adaptive_controller import AdaptiveBEDSController
from knowswe.beds.chunk_environment import ChunkInvestigationEnvironment
from knowswe.beds.hypotheses import HypothesisTracker


EvidenceEvaluator = Callable[
    [str, str, str, tuple[str, ...]],
    dict[str, float],
]


@dataclass
class HypothesisInvestigationResult:
    state: object
    hypotheses: HypothesisTracker


def investigate_with_hypotheses(
    repository_path: str,
    bug_report: str,
    evaluator: EvidenceEvaluator,
    budget: int = 2000,
    top_k: int = 5,
    max_steps: int = 10,
    max_chunk_tokens: int = 300,
    relevance_mode: str = "chunk",
) -> HypothesisInvestigationResult:

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

    for _ in range(max_steps):

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

            controller.observe(
                selected,
                observation.content,
                actual_cost=actual_cost,
            )

        except Exception:
            controller.cancel_action()
            raise

        pending_targets = tuple(
            action.target
            for action in controller.state.pending_actions.values()
        )

        updates = evaluator(
            bug_report,
            selected.target,
            observation.content,
            pending_targets,
        )

        for location, support in updates.items():
            if location not in tracker.hypotheses:
                continue

            tracker.update(
                location=location,
                evidence_id=selected.target,
                support=support,
            )

        # Rebuild pending actions using updated hypothesis scores.
        updated_actions = {}

        for key, action in controller.state.pending_actions.items():

            hypothesis = tracker.hypotheses.get(action.target)

            confidence = (
                hypothesis.confidence
                if hypothesis is not None
                else action.relevance
            )

            updated_actions[key] = InvestigationAction(
                action_type=action.action_type,
                target=action.target,
                estimated_cost=action.estimated_cost,
                relevance=confidence,
                diagnostic_value=action.diagnostic_value,
                information_gain_proxy=action.information_gain_proxy,
            )

        controller.state.pending_actions = updated_actions

    return HypothesisInvestigationResult(
        state=controller.state,
        hypotheses=tracker,
    )
