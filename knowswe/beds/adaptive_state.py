from dataclasses import dataclass, field

from knowswe.beds.adaptive_actions import InvestigationAction


@dataclass
class Evidence:
    action_key: tuple[str, str]
    content: str
    discovered_targets: list[str] = field(
        default_factory=list
    )


@dataclass
class InvestigationState:
    initial_budget: int
    remaining_budget: int

    evidence: list[Evidence] = field(default_factory=list)

    completed_actions: set[tuple[str, str]] = field(
        default_factory=set
    )

    pending_actions: dict[
        tuple[str, str], InvestigationAction
    ] = field(default_factory=dict)

    history: list[dict] = field(default_factory=list)

    active_action: tuple[str, str] | None = None
    stop_reason: str | None = None

    @property
    def spent_budget(self):
        return self.initial_budget - self.remaining_budget
