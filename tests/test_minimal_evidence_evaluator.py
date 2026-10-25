import unittest

from knowswe.models.minimal_evidence_evaluator import (
    MinimalGemmaEvidenceEvaluator,
)


class FakeClient:
    def __init__(self, response):
        self.response = response

    def generate_json(self, prompt):
        return self.response


class TestMinimalEvidenceEvaluator(unittest.TestCase):

    def setUp(self):
        self.location = "src/example.py:10:11"
        self.content = "x = 1\nreturn x + 1\n"

    def assess(self, response):
        evaluator = MinimalGemmaEvidenceEvaluator(
            FakeClient(response)
        )

        return evaluator.assess(
            bug_report="Incorrect calculation",
            location=self.location,
            content=self.content,
            start_line=10,
        )

    def test_positive_support(self):
        result = self.assess({
            "support": 1,
            "line": 11,
        })

        self.assertEqual(result.support, 1)
        self.assertEqual(result.evidence_line, 11)
        self.assertEqual(result.evidence_text, "return x + 1")

    def test_negative_support(self):
        result = self.assess({
            "support": -1,
            "line": 10,
        })

        self.assertEqual(result.support, -1)
        self.assertEqual(result.evidence_line, 10)

    def test_neutral_support(self):
        result = self.assess({
            "support": 0,
            "line": None,
        })

        self.assertEqual(result.support, 0)
        self.assertIsNone(result.evidence_text)

    def test_rejects_invalid_line(self):
        with self.assertRaises(ValueError):
            self.assess({
                "support": 1,
                "line": 99,
            })

    def test_rejects_missing_line(self):
        with self.assertRaises(ValueError):
            self.assess({
                "support": 1,
                "line": None,
            })

    def test_rejects_boolean_support(self):
        with self.assertRaises(ValueError):
            self.assess({
                "support": True,
                "line": 10,
            })

    def test_rejects_extra_fields(self):
        with self.assertRaises(ValueError):
            self.assess({
                "support": 1,
                "line": 10,
                "reason": "Unexpected field",
            })

    def test_rejects_invalid_support(self):
        with self.assertRaises(ValueError):
            self.assess({
                "support": 2,
                "line": 10,
            })


if __name__ == "__main__":
    unittest.main()
