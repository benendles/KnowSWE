"""Run a budgeted chunk-level repository investigation."""

from knowswe.beds.adaptive_controller import (
    AdaptiveBEDSController,
)
from knowswe.beds.chunk_environment import (
    ChunkInvestigationEnvironment,
)


def investigate_repository_chunks(
    repository_path: str,
    bug_report: str,
    budget: int = 2000,
    top_k: int = 5,
    max_steps: int = 10,
    max_chunk_tokens: int = 300,
    relevance_mode: str = 'file',
):
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
                discovered_actions=observation.discovered_actions,
                actual_cost=actual_cost,
            )

        except Exception:
            controller.cancel_action()
            raise

    return controller.state
