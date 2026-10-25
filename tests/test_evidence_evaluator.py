import unittest

from knowswe.models.evidence_evaluator import (
    GemmaEvidenceEvaluator,
)


class FakeClient:
    def __init__(self, response):
        self.response = response

    def generate_json(self, prompt):
        return self.response


class TestEvidenceEvaluator(unittest.TestCase):

    def setUp(self):
        self.location = "src/example.py:1:3"
        self.content = "def calculate(x):\n    return x + 1\n"

    def evaluate(self, response):
        evaluator = GemmaEvidenceEvaluator(FakeClient(response))

        return evaluator.assess(
            bug_report="Calculation produces incorrect results",
            location=self.location,
            content=self.content,
        )

    def valid_response(self):
        return {
            "location": self.location,
            "support": 0.7,
            "evidence_quote": "return x + 1",
            "reason": "The calculation may be incorrect.",
        }

    def test_valid_assessment(self):
        result = self.evaluate(self.valid_response())

        self.assertEqual(result.support, 0.7)
        self.assertEqual(result.location, self.location)

    def test_rejects_wrong_location(self):
        response = self.valid_response()
        response["location"] = "another.py"

        with self.assertRaises(ValueError):
            self.evaluate(response)

    def test_rejects_fabricated_quote(self):
        response = self.valid_response()
        response["evidence_quote"] = "return x * 100"

        with self.assertRaises(ValueError):
            self.evaluate(response)

    def test_rejects_support_without_quote(self):
        response = self.valid_response()
        response["evidence_quote"] = ""

        with self.assertRaises(ValueError):
            self.evaluate(response)

    def test_rejects_out_of_range_support(self):
        response = self.valid_response()
        response["support"] = 2.0

        with self.assertRaises(ValueError):
            self.evaluate(response)

    def test_accepts_neutral_assessment(self):
        response = self.valid_response()
        response["support"] = 0
        response["evidence_quote"] = ""

        result = self.evaluate(response)

        self.assertEqual(result.support, 0.0)

    def test_rejects_boolean_support(self):
        response = self.valid_response()
        response["support"] = True

        with self.assertRaises(ValueError):
            self.evaluate(response)


if __name__ == "__main__":
    unittest.main()
