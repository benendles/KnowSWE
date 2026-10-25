import tempfile
import unittest
from pathlib import Path

from knowswe.beds.gemma_runner import investigate_with_gemma
from knowswe.models.minimal_evidence_evaluator import (
    MinimalEvidenceAssessment,
)


class FakeEvaluator:
    def __init__(self, support=1):
        self.support = support
        self.calls = []

    def assess(self, bug_report, location, content, start_line):
        self.calls.append(location)

        return MinimalEvidenceAssessment(
            location=location,
            support=self.support,
            evidence_line=start_line if self.support else None,
            evidence_text=content.splitlines()[0] if self.support else None,
        )


class TestGemmaRunner(unittest.TestCase):

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

    def run_investigation(self, evaluator, **kwargs):
        return investigate_with_gemma(
            repository_path=str(self.repo),
            bug_report="calculate_total returns incorrect total",
            evaluator=evaluator,
            **kwargs,
        )

    def test_records_validated_assessment(self):
        evaluator = FakeEvaluator(support=1)

        result = self.run_investigation(
            evaluator,
            budget=500,
            max_steps=1,
        )

        self.assertEqual(len(result.assessments), 1)

        assessment = result.assessments[0]

        self.assertEqual(assessment.support, 1)
        self.assertEqual(assessment.step, 1)
        self.assertIsNotNone(assessment.evidence_line)
        self.assertGreaterEqual(assessment.inference_seconds, 0)

    def test_respects_inspection_budget(self):
        evaluator = FakeEvaluator()

        result = self.run_investigation(
            evaluator,
            budget=500,
            max_steps=10,
        )

        self.assertLessEqual(result.state.spent_budget, 500)

        self.assertEqual(
            len(result.assessments),
            len(result.state.history),
        )

    def test_no_duplicate_inspections(self):
        evaluator = FakeEvaluator()

        result = self.run_investigation(
            evaluator,
            budget=500,
            max_steps=10,
        )

        locations = [
            assessment.location
            for assessment in result.assessments
        ]

        self.assertEqual(
            len(locations),
            len(set(locations)),
        )

    def test_evaluator_failure_does_not_commit_inspection(self):
        class FailingEvaluator:
            def assess(self, **kwargs):
                raise ValueError("Model returned invalid JSON")

        with self.assertRaisesRegex(
            ValueError,
            "Model returned invalid JSON",
        ):
            self.run_investigation(
                FailingEvaluator(),
                budget=500,
                max_steps=1,
            )

    def test_rejects_invalid_assessment_support(self):
        evaluator = FakeEvaluator(support=2)

        with self.assertRaisesRegex(
            ValueError,
            "Invalid assessment support",
        ):
            self.run_investigation(
                evaluator,
                budget=500,
                max_steps=1,
            )


if __name__ == "__main__":
    unittest.main()
