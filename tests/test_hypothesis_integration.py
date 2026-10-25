import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from knowswe.beds.adaptive_actions import (
    ActionType,
    InvestigationAction,
)
from knowswe.beds.hypothesis_runner import investigate_with_hypotheses


@dataclass
class FakeChunk:
    estimated_tokens: int = 50


class FakeEnvironment:
    def __init__(self, *args, **kwargs):
        self.actions = [
            InvestigationAction(
                action_type=ActionType.INSPECT,
                target=target,
                estimated_cost=50,
                relevance=0.5,
                diagnostic_value=0.5,
            )
            for target in (
                "a.py:1:10",
                "b.py:1:10",
                "c.py:1:10",
            )
        ]

    def initial_actions(self, bug_report, top_k=5):
        return self.actions

    def execute(self, action):
        return SimpleNamespace(
            content=f"Source code for {action.target}",
            chunk=FakeChunk(),
        )


class TestHypothesisIntegration(unittest.TestCase):

    def run_case(self, supported_target):
        calls = []

        def evaluator(report, target, content, pending):
            calls.append(target)

            if len(calls) == 1:
                return {supported_target: 1.0}

            return {}

        with patch(
            "knowswe.beds.hypothesis_runner."
            "ChunkInvestigationEnvironment",
            FakeEnvironment,
        ):
            result = investigate_with_hypotheses(
                repository_path="unused",
                bug_report="Example bug",
                evaluator=evaluator,
                budget=300,
                max_steps=2,
            )

        return [
            entry["target"]
            for entry in result.state.history
        ]

    def test_evidence_changes_second_inspection(self):
        history_a = self.run_case("a.py:1:10")
        history_b = self.run_case("b.py:1:10")

        self.assertEqual(history_a[0], history_b[0])

        self.assertEqual(history_a[1], "a.py:1:10")
        self.assertEqual(history_b[1], "b.py:1:10")

        self.assertNotEqual(history_a[1], history_b[1])


if __name__ == "__main__":
    unittest.main()
