from knowswe.beds.adaptive_controller import (
    AdaptiveBEDSController,
)
from knowswe.beds.repository_environment import (
    RepositoryInvestigationEnvironment,
)


def investigate_repository(
    repository_path: str,
    bug_report: str,
    budget: int = 2000,
    top_k: int = 5,
    max_steps: int = 10,
):
    environment = RepositoryInvestigationEnvironment(
        repository_path
    )

    controller = AdaptiveBEDSController(
        budget=budget
    )

    controller.add_actions(
        environment.initial_actions(
            bug_report,
            top_k=top_k,
        )
    )

    for _ in range(max_steps):
        selected = controller.select_next()

        if selected is None:
            break

        observation = environment.execute(selected)

        controller.observe(
            selected,
            observation.content,
            discovered_actions=observation.discovered_actions,
        )

    return controller.state
