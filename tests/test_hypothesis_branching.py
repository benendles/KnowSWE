import unittest
from dataclasses import replace

from knowswe.beds.adaptive_actions import (
    ActionType,
    InvestigationAction,
)
from knowswe.beds.adaptive_controller import AdaptiveBEDSController
from knowswe.beds.hypotheses import HypothesisTracker


class TestHypothesisBranching(unittest.TestCase):

    def run_investigation(self, supported_target):
        controller = AdaptiveBEDSController(
            budget=300,
            cost_exponent=0.0,
        )

        actions = [
            InvestigationAction(
                action_type=ActionType.INSPECT,
                target=name,
                estimated_cost=50,
                relevance=0.5,
                diagnostic_value=0.5,
            )
            for name in (
                "a.py:1:10",
                "b.py:1:10",
                "c.py:1:10",
            )
        ]

        controller.add_actions(actions)

        tracker = HypothesisTracker()

        for action in actions:
            tracker.add(action.target, confidence=0.5)

        first = controller.select_next()
        controller.begin_action(first)

        controller.observe(
            first,
            content="Observed source-code evidence",
            actual_cost=50,
        )

        remaining = [
            action.target
            for action in controller.state.pending_actions.values()
        ]

        self.assertEqual(len(remaining), 2)
        self.assertIn(supported_target, remaining)

        tracker.update(
            location=supported_target,
            evidence_id=first.target,
            support=1.0,
        )

        controller.state.pending_actions = {
            key: replace(
                action,
                relevance=tracker.hypotheses[
                    action.target
                ].confidence,
            )
            for key, action
            in controller.state.pending_actions.items()
        }

        second = controller.select_next()

        return first.target, second.target

    def test_evidence_changes_second_action(self):
        first_a, second_a = self.run_investigation(
            "a.py:1:10"
        )

        first_b, second_b = self.run_investigation(
            "b.py:1:10"
        )

        self.assertEqual(first_a, first_b)
        self.assertEqual(second_a, "a.py:1:10")
        self.assertEqual(second_b, "b.py:1:10")
        self.assertNotEqual(second_a, second_b)


if __name__ == "__main__":
    unittest.main()
