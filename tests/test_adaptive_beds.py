import unittest

from knowswe.beds.adaptive_actions import (
    ActionType,
    InvestigationAction,
)
from knowswe.beds.adaptive_controller import (
    AdaptiveBEDSController,
)


def action(
    target,
    cost,
    relevance=0.5,
    diagnostic=0.5,
    kind=ActionType.INSPECT,
):
    return InvestigationAction(
        action_type=kind,
        target=target,
        estimated_cost=cost,
        relevance=relevance,
        diagnostic_value=diagnostic,
    )


class TestAdaptiveBEDS(unittest.TestCase):

    def test_budget_is_respected(self):
        controller = AdaptiveBEDSController(budget=100)

        controller.add_actions([
            action("a.py", 70),
            action("b.py", 60),
        ])

        first = controller.select_next()

        controller.begin_action(first)
        controller.observe(first, "Evidence")

        self.assertLessEqual(
            controller.state.spent_budget,
            100
        )

        self.assertIsNone(
            controller.select_next()
        )

    def test_completed_actions_not_repeated(self):
        controller = AdaptiveBEDSController(budget=100)

        candidate = action("a.py", 20)

        controller.add_actions([candidate])

        selected = controller.select_next()

        controller.begin_action(selected)
        controller.observe(selected, "Evidence")

        self.assertNotIn(
            candidate.key,
            controller.state.pending_actions
        )

        controller.add_actions([candidate])

        self.assertIsNone(
            controller.select_next()
        )

    def test_new_evidence_changes_next_action(self):
        controller = AdaptiveBEDSController(budget=200)

        initial = action(
            "entry.py",
            20,
            relevance=0.9,
            diagnostic=0.9,
        )

        alternative = action(
            "unrelated.py",
            20,
            relevance=0.1,
            diagnostic=0.1,
        )

        controller.add_actions([
            initial,
            alternative,
        ])

        selected = controller.select_next()

        self.assertEqual(
            selected.target,
            "entry.py"
        )

        discovered = action(
            "dependency.py",
            20,
            relevance=0.95,
            diagnostic=0.95,
            kind=ActionType.FOLLOW_DEPENDENCY,
        )

        controller.begin_action(selected)

        controller.observe(
            selected,
            "entry.py calls dependency.py",
            discovered_actions=[discovered],
        )

        next_action = controller.select_next()

        self.assertEqual(
            next_action.target,
            "dependency.py"
        )

    def test_exhausted_budget_stops(self):
        controller = AdaptiveBEDSController(budget=20)

        candidate = action("a.py", 20)

        controller.add_actions([candidate])

        selected = controller.select_next()
        controller.begin_action(selected)

        controller.observe(
            selected,
            "Evidence",
        )

        self.assertTrue(
            controller.should_stop()
        )


    def test_unauthorized_observation_rejected(self):
        controller = AdaptiveBEDSController(budget=100)

        candidate = action("a.py", 20)

        controller.add_actions([candidate])

        with self.assertRaises(ValueError):
            controller.observe(candidate, "Evidence")

        self.assertEqual(
            controller.state.remaining_budget,
            100,
        )

    def test_actual_cost_is_recorded(self):
        controller = AdaptiveBEDSController(budget=100)

        candidate = action("a.py", 20)

        controller.add_actions([candidate])
        controller.begin_action(candidate)

        controller.observe(
            candidate,
            "Evidence",
            actual_cost=35,
        )

        self.assertEqual(
            controller.state.spent_budget,
            35,
        )


    def test_same_file_not_inspected_twice(self):
        controller = AdaptiveBEDSController(budget=100)

        first = action(
            "helper.py",
            20,
            kind=ActionType.INSPECT,
        )

        duplicate = action(
            "helper.py",
            20,
            kind=ActionType.FOLLOW_DEPENDENCY,
        )

        controller.add_actions([first])
        controller.begin_action(first)
        controller.observe(first, "Evidence")

        controller.add_actions([duplicate])

        self.assertIsNone(controller.select_next())

        self.assertEqual(len(controller.state.history), 1)


if __name__ == "__main__":
    unittest.main()
