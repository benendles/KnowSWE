import unittest

from knowswe.beds.hypotheses import HypothesisTracker


class TestHypothesisTracker(unittest.TestCase):

    def test_add_hypothesis(self):
        tracker = HypothesisTracker()
        tracker.add("module.py:10", 0.5)

        self.assertEqual(
            tracker.hypotheses["module.py:10"].confidence,
            0.5,
        )

    def test_support_increases_confidence(self):
        tracker = HypothesisTracker()
        tracker.add("module.py:10", 0.5)

        tracker.update(
            "module.py:10",
            evidence_id="inspection-1",
            support=1.0,
        )

        self.assertAlmostEqual(
            tracker.hypotheses["module.py:10"].confidence,
            0.7,
        )

    def test_contradiction_decreases_confidence(self):
        tracker = HypothesisTracker()
        tracker.add("module.py:10", 0.5)

        tracker.update(
            "module.py:10",
            evidence_id="inspection-1",
            support=-1.0,
        )

        self.assertAlmostEqual(
            tracker.hypotheses["module.py:10"].confidence,
            0.3,
        )

    def test_confidence_is_bounded(self):
        tracker = HypothesisTracker()
        tracker.add("module.py:10", 0.95)

        tracker.update(
            "module.py:10",
            evidence_id="inspection-1",
            support=1.0,
        )

        self.assertEqual(
            tracker.hypotheses["module.py:10"].confidence,
            1.0,
        )

    def test_ranking(self):
        tracker = HypothesisTracker()

        tracker.add("a.py:10", 0.3)
        tracker.add("b.py:20", 0.8)

        ranked = tracker.ranked()

        self.assertEqual(ranked[0].location, "b.py:20")


if __name__ == "__main__":
    unittest.main()
