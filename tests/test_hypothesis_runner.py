import tempfile
import unittest
from pathlib import Path

from knowswe.beds.hypothesis_runner import (
    investigate_with_hypotheses,
)


class TestHypothesisRunner(unittest.TestCase):

    def make_repository(self, directory):
        root = Path(directory)

        (root / "example.py").write_text(
            "def process_payment(amount):\n"
            "    return amount * 2\n"
            "\n"
            "def validate_invoice(invoice):\n"
            "    return invoice is not None\n"
            "\n"
            "def calculate_tax(amount):\n"
            "    return amount * 0.2\n"
        )

    def test_budget_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_repository(directory)

            def evaluator(report, target, content, pending):
                return {}

            result = investigate_with_hypotheses(
                repository_path=directory,
                bug_report="payment invoice tax",
                evaluator=evaluator,
                budget=100,
                top_k=1,
                max_steps=3,
                max_chunk_tokens=100,
            )

            self.assertLessEqual(
                result.state.spent_budget,
                100,
            )

    def test_evidence_updates_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_repository(directory)

            def evaluator(report, target, content, pending):
                return {
                    location: 1.0
                    for location in pending
                }

            result = investigate_with_hypotheses(
                repository_path=directory,
                bug_report="payment invoice tax",
                evaluator=evaluator,
                budget=100,
                top_k=1,
                max_steps=1,
                max_chunk_tokens=100,
            )

            self.assertTrue(
                any(
                    hypothesis.supporting_evidence
                    for hypothesis
                    in result.hypotheses.hypotheses.values()
                )
            )

    def test_no_duplicate_inspections(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_repository(directory)

            result = investigate_with_hypotheses(
                repository_path=directory,
                bug_report="payment invoice tax",
                evaluator=lambda *args: {},
                budget=100,
                top_k=1,
                max_steps=10,
                max_chunk_tokens=100,
            )

            targets = [
                entry["target"]
                for entry in result.state.history
            ]

            self.assertEqual(
                len(targets),
                len(set(targets)),
            )


if __name__ == "__main__":
    unittest.main()
