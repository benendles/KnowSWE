import tempfile
import unittest
from pathlib import Path

from knowswe.beds.resilient_gemma_runner import (
    investigate_resiliently,
)
from knowswe.models.minimal_evidence_evaluator import (
    MinimalEvidenceAssessment,
)
from knowswe.models.resilient_evaluator import (
    ResilientEvidenceEvaluator,
)


class FakeEvaluator:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0

    def assess(self, bug_report, location, content, start_line):
        self.calls += 1

        if self.calls <= self.failures:
            raise ValueError("Invalid model response")

        return MinimalEvidenceAssessment(
            location=location,
            support=1,
            evidence_line=start_line,
            evidence_text=content.splitlines()[0],
        )


class TestResilientGemmaRunner(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.repo = Path(self.temp.name)

        (self.repo / "calculator.py").write_text(
            "def calculate_total(price, quantity):\n"
            "    return price * quantity\n"
            "\n"
            "def calculate_discount(price, discount):\n"
            "    return price - discount\n"
        )

    def investigate(self, evaluator, **kwargs):
        return investigate_resiliently(
            repository_path=str(self.repo),
            bug_report="calculate_total returns incorrect total",
            evaluator=evaluator,
            budget=500,
            **kwargs,
        )

    def test_successful_assessment(self):
        evaluator = ResilientEvidenceEvaluator(
            FakeEvaluator(),
            max_retries=1,
        )

        result = self.investigate(evaluator, max_steps=1)

        self.assertEqual(len(result.steps), 1)
        self.assertTrue(result.steps[0].success)
        self.assertEqual(result.steps[0].support, 1)

    def test_success_after_retry(self):
        evaluator = ResilientEvidenceEvaluator(
            FakeEvaluator(failures=1),
            max_retries=1,
        )

        result = self.investigate(evaluator, max_steps=1)

        self.assertTrue(result.steps[0].success)
        self.assertEqual(result.steps[0].inference_attempts, 2)

    def test_failure_does_not_change_confidence(self):
        evaluator = ResilientEvidenceEvaluator(
            FakeEvaluator(failures=100),
            max_retries=1,
        )

        result = self.investigate(evaluator, max_steps=1)

        step = result.steps[0]

        self.assertFalse(step.success)
        self.assertIsNone(step.support)

        self.assertEqual(
            step.confidence_before,
            step.confidence_after,
        )

        self.assertEqual(step.inference_attempts, 2)

    def test_continues_after_failed_assessment(self):
        evaluator = ResilientEvidenceEvaluator(
            FakeEvaluator(failures=2),
            max_retries=1,
        )

        result = self.investigate(evaluator, max_steps=2)

        self.assertEqual(len(result.steps), 2)
        self.assertFalse(result.steps[0].success)
        self.assertTrue(result.steps[1].success)

    def test_inspection_budget_respected(self):
        evaluator = ResilientEvidenceEvaluator(
            FakeEvaluator(),
            max_retries=0,
        )

        result = self.investigate(evaluator, max_steps=10)

        self.assertLessEqual(result.state.spent_budget, 500)

        self.assertEqual(
            len(result.steps),
            len(result.state.history),
        )


if __name__ == "__main__":
    unittest.main()
