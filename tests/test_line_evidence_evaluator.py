import unittest

from knowswe.models.line_evidence_evaluator import (
    GemmaLineEvidenceEvaluator,
)


class FakeClient:
    def __init__(self, response):
        self.response = response

    def generate_json(self, prompt):
        return self.response


class TestLineEvidenceEvaluator(unittest.TestCase):

    def setUp(self):
        self.location = "src/example.py:10:11"
        self.content = "x = 1\nreturn x + 1\n"

    def response(self):
        return {
            "location": self.location,
            "support": 0.5,
            "evidence_lines": [11],
            "reason": "The calculation may be incorrect.",
        }

    def assess(self, response):
        evaluator = GemmaLineEvidenceEvaluator(
            FakeClient(response)
        )

        return evaluator.assess(
            bug_report="Incorrect calculation",
            location=self.location,
            content=self.content,
            start_line=10,
        )

    def test_valid_assessment(self):
        result = self.assess(self.response())

        self.assertEqual(result.support, 0.5)
        self.assertEqual(result.evidence_lines, (11,))
        self.assertEqual(result.evidence_text, "11: return x + 1")

    def test_rejects_unknown_line(self):
        response = self.response()
        response["evidence_lines"] = [99]

        with self.assertRaises(ValueError):
            self.assess(response)

    def test_rejects_missing_reason(self):
        response = self.response()
        del response["reason"]

        with self.assertRaises(ValueError):
            self.assess(response)

    def test_rejects_support_without_evidence(self):
        response = self.response()
        response["evidence_lines"] = []

        with self.assertRaises(ValueError):
            self.assess(response)

    def test_accepts_neutral_assessment(self):
        response = self.response()
        response["support"] = 0
        response["evidence_lines"] = []

        result = self.assess(response)

        self.assertEqual(result.support, 0)

    def test_rejects_boolean_line(self):
        response = self.response()
        response["evidence_lines"] = [True]

        with self.assertRaises(ValueError):
            self.assess(response)

    def test_rejects_duplicate_lines(self):
        response = self.response()
        response["evidence_lines"] = [11, 11]

        with self.assertRaises(ValueError):
            self.assess(response)


if __name__ == "__main__":
    unittest.main()
