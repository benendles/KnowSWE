from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    RETRIEVE = "retrieve"
    INSPECT = "inspect"
    FOLLOW_DEPENDENCY = "follow_dependency"
    INSPECT_TEST = "inspect_test"


@dataclass(frozen=True)
class InvestigationAction:
    action_type: ActionType
    target: str
    estimated_cost: int
    relevance: float = 0.0
    diagnostic_value: float = 0.0
    information_gain_proxy: float = 0.0

    @property
    def key(self):
        return (self.action_type.value, self.target)
