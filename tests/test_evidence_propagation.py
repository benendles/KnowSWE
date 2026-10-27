import unittest

from knowswe.beds.evidence_propagation import (
    propagate_evidence,
    structural_relationship,
)
from knowswe.chunking.method_chunker import MethodChunk


def chunk(filename, start, symbol, parent=None):
    return MethodChunk(
        filename=filename,
        start_line=start,
        end_line=start + 5,
        symbol=symbol,
        parent_symbol=parent,
        content="def example():\n    return 1\n",
        estimated_tokens=10,
    )


class TestEvidencePropagation(unittest.TestCase):

    def setUp(self):
        self.source = chunk("module.py", 10, "Worker.process", "Worker")
        self.same_method = chunk("module.py", 20, "Worker.process", "Worker")
        self.same_class = chunk("module.py", 30, "Worker.validate", "Worker")
        self.same_file = chunk("module.py", 40, "other_function")
        self.other_file = chunk("another.py", 50, "Worker.process", "Worker")

    def test_same_method(self):
        self.assertEqual(
            structural_relationship(self.source, self.same_method),
            ("same_method", 1.0),
        )

    def test_same_class(self):
        self.assertEqual(
            structural_relationship(self.source, self.same_class),
            ("same_class", 0.5),
        )

    def test_same_file(self):
        self.assertEqual(
            structural_relationship(self.source, self.same_file),
            ("same_file", 0.15),
        )

    def test_different_file(self):
        self.assertEqual(
            structural_relationship(self.source, self.other_file),
            ("different_file", 0.0),
        )

    def test_positive_propagation(self):
        results = propagate_evidence(
            self.source,
            [self.same_method, self.same_class, self.same_file, self.other_file],
            support=1,
        )

        deltas = {result.target: result.confidence_delta for result in results}

        self.assertAlmostEqual(deltas[self.same_method.chunk_id], 0.2)
        self.assertAlmostEqual(deltas[self.same_class.chunk_id], 0.1)
        self.assertAlmostEqual(deltas[self.same_file.chunk_id], 0.03)
        self.assertNotIn(self.other_file.chunk_id, deltas)

    def test_negative_propagation(self):
        results = propagate_evidence(
            self.source,
            [self.same_method],
            support=-1,
        )

        self.assertAlmostEqual(results[0].confidence_delta, -0.2)

    def test_neutral_propagation(self):
        results = propagate_evidence(
            self.source,
            [self.same_method],
            support=0,
        )

        self.assertEqual(results, [])

    def test_invalid_support(self):
        with self.assertRaises(ValueError):
            propagate_evidence(self.source, [self.same_method], support=2)


if __name__ == "__main__":
    unittest.main()
