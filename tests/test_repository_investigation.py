import tempfile
import unittest
from pathlib import Path

from knowswe.beds.adaptive_actions import (
    ActionType,
    InvestigationAction,
)
from knowswe.beds.adaptive_controller import (
    AdaptiveBEDSController,
)
from knowswe.beds.repository_environment import (
    RepositoryInvestigationEnvironment,
)


class TestRepositoryInvestigation(unittest.TestCase):

    def test_discovers_dependency_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (root / "main.py").write_text(
                "from helper import calculate_total\n\n"
                "def process_invoice():\n"
                "    return calculate_total(10, 20)\n"
            )

            (root / "helper.py").write_text(
                "def calculate_total(a, b):\n"
                "    return a + b\n"
            )

            environment = RepositoryInvestigationEnvironment(
                directory
            )

            controller = AdaptiveBEDSController(
                budget=1000
            )

            initial = InvestigationAction(
                action_type=ActionType.INSPECT,
                target="main.py",
                estimated_cost=100,
                relevance=1.0,
                diagnostic_value=1.0,
            )

            controller.add_actions([initial])

            first = controller.select_next()

            self.assertEqual(
                first.target,
                "main.py"
            )

            observation = environment.execute(first)

            discovered = {
                action.target
                for action in observation.discovered_actions
            }

            self.assertIn(
                "helper.py",
                discovered
            )

            controller.observe(
                first,
                observation.content,
                discovered_actions=observation.discovered_actions,
            )

            second = controller.select_next()

            self.assertIsNotNone(second)

            self.assertEqual(
                second.target,
                "helper.py"
            )

            second_observation = environment.execute(second)

            controller.observe(
                second,
                second_observation.content,
                discovered_actions=second_observation.discovered_actions,
            )

            inspected = {
                entry["target"]
                for entry in controller.state.history
            }

            self.assertEqual(
                inspected,
                {"main.py", "helper.py"}
            )

            self.assertLessEqual(
                controller.state.spent_budget,
                1000
            )


if __name__ == "__main__":
    unittest.main()
