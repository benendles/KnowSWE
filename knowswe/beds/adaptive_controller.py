from knowswe.beds.adaptive_actions import InvestigationAction
from knowswe.beds.adaptive_state import (
    Evidence,
    InvestigationState,
)


class AdaptiveBEDSController:
    """
    Stateful, budget-constrained investigation controller.

    BEDS-1 supports adaptive action selection from externally
    supplied candidate actions and observations.

    It does not yet perform model-based hypothesis updates.
    """

    def __init__(
        self,
        budget: int = 2000,
        relevance_weight: float = 0.4,
        diagnostic_weight: float = 0.4,
        information_weight: float = 0.2,
        cost_exponent: float = 0.5,
    ):
        if budget <= 0:
            raise ValueError("Budget must be positive")

        if cost_exponent < 0:
            raise ValueError(
                "Cost exponent cannot be negative"
            )

        self.state = InvestigationState(
            initial_budget=budget,
            remaining_budget=budget,
        )

        self.relevance_weight = relevance_weight
        self.diagnostic_weight = diagnostic_weight
        self.information_weight = information_weight
        self.cost_exponent = cost_exponent

    def add_actions(self, actions):
        for action in actions:
            if not isinstance(action, InvestigationAction):
                raise TypeError(
                    "Expected InvestigationAction"
                )

            if action.estimated_cost <= 0:
                continue

            if action.key in self.state.completed_actions:
                continue

            self.state.pending_actions[action.key] = action

    def score(self, action):
        relevance = max(action.relevance, 0.0)
        diagnostic = max(action.diagnostic_value, 0.0)
        information = max(
            action.information_gain_proxy, 0.0
        )

        numerator = (
            self.relevance_weight * relevance
            + self.diagnostic_weight * diagnostic
            + self.information_weight * information
        )

        denominator = (
            action.estimated_cost + 1
        ) ** self.cost_exponent

        return numerator / denominator

    def select_next(self):
        available = [
            action
            for action in self.state.pending_actions.values()
            if action.estimated_cost
            <= self.state.remaining_budget
        ]

        if not available:
            return None

        return max(
            available,
            key=lambda action: (
                self.score(action),
                -action.estimated_cost,
                action.key,
            ),
        )

    def observe(
        self,
        action: InvestigationAction,
        content: str,
        discovered_actions=None,
    ):
        if action.key not in self.state.pending_actions:
            raise ValueError(
                "Action was not registered"
            )

        if action.key in self.state.completed_actions:
            raise ValueError(
                "Action already completed"
            )

        if action.estimated_cost > self.state.remaining_budget:
            raise ValueError(
                "Insufficient investigation budget"
            )

        self.state.remaining_budget -= action.estimated_cost

        self.state.completed_actions.add(action.key)

        self.state.pending_actions.pop(
            action.key, None
        )

        discovered_actions = discovered_actions or []

        evidence = Evidence(
            action_key=action.key,
            content=content,
            discovered_targets=[
                candidate.target
                for candidate in discovered_actions
            ],
        )

        self.state.evidence.append(evidence)

        self.state.history.append({
            "step": len(self.state.history) + 1,
            "action": action.action_type.value,
            "target": action.target,
            "cost": action.estimated_cost,
            "remaining_budget": self.state.remaining_budget,
            "new_actions": len(discovered_actions),
        })

        self.add_actions(discovered_actions)

        return evidence

    def should_stop(self):
        return self.select_next() is None
