import unittest

from knowswe.models.resilient_evaluator import (
    ResilientEvidenceEvaluator,
)


class FakeEvaluator:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0

    def assess(self, **kwargs):
        self.calls += 1

        if self.calls <= self.failures:
            raise ValueError("Invalid model output")

        return {"support": 1, "line": 10}


class TestResilientEvaluator(unittest.TestCase):

    def test_success_without_retry(self):
        evaluator = FakeEvaluator()
        result = ResilientEvidenceEvaluator(
            evaluator,
            max_retries=1,
        ).assess()

        self.assertTrue(result.success)
        self.assertEqual(len(result.attempts), 1)

    def test_success_after_retry(self):
        evaluator = FakeEvaluator(failures=1)

        result = ResilientEvidenceEvaluator(
            evaluator,
            max_retries=1,
        ).assess()

        self.assertTrue(result.success)
        self.assertEqual(len(result.attempts), 2)
        self.assertFalse(result.attempts[0].success)
        self.assertTrue(result.attempts[1].success)

    def test_failure_after_retries(self):
        evaluator = FakeEvaluator(failures=3)

        result = ResilientEvidenceEvaluator(
            evaluator,
            max_retries=1,
        ).assess()

        self.assertFalse(result.success)
        self.assertIsNone(result.assessment)
        self.assertEqual(len(result.attempts), 2)

    def test_no_retries(self):
        evaluator = FakeEvaluator(failures=1)

        result = ResilientEvidenceEvaluator(
            evaluator,
            max_retries=0,
        ).assess()

        self.assertFalse(result.success)
        self.assertEqual(evaluator.calls, 1)

    def test_rejects_negative_retries(self):
        with self.assertRaises(ValueError):
            ResilientEvidenceEvaluator(
                FakeEvaluator(),
                max_retries=-1,
            )


if __name__ == "__main__":
    unittest.main()
