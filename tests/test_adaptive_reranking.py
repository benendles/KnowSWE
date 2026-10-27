import unittest

from knowswe.beds.adaptive_actions import (
    ActionType,
    InvestigationAction,
)
from knowswe.beds.adaptive_controller import AdaptiveBEDSController
from knowswe.beds.adaptive_reranking import rerank_pending_actions
from knowswe.beds.hypotheses import HypothesisTracker
from knowswe.chunking.method_chunker import MethodChunk


def make_chunk(filename, start, symbol, parent=None):
    return MethodChunk(
        filename=filename,
        start_line=start,
        end_line=start + 4,
        symbol=symbol,
        parent_symbol=parent,
        content="def example():\n    return 1\n",
        estimated_tokens=20,
    )


def make_action(chunk, relevance):
    return InvestigationAction(
        action_type=ActionType.INSPECT,
        target=chunk.chunk_id,
        estimated_cost=20,
        relevance=relevance,
        diagnostic_value=0.5,
    )


class TestAdaptiveReranking(unittest.TestCase):

    def setUp(self):
        self.source = make_chunk(
            "worker.py", 10, "Worker.process", "Worker"
        )

        self.related = make_chunk(
            "worker.py", 20, "Worker.process", "Worker"
        )

        self.unrelated = make_chunk(
            "other.py", 30, "Other.run", "Other"
        )

        self.chunks = {
            chunk.chunk_id: chunk
            for chunk in (
                self.source,
                self.related,
                self.unrelated,
            )
        }

        self.controller = AdaptiveBEDSController(
            budget=200,
            cost_exponent=0.0,
        )

        self.tracker = HypothesisTracker()

        self.actions = [
            make_action(self.source, 0.9),
            make_action(self.related, 0.4),
            make_action(self.unrelated, 0.5),
        ]

        self.controller.add_actions(self.actions)

        for action in self.actions:
            self.tracker.add(
                action.target,
                confidence=action.relevance,
            )

        first = self.controller.select_next()

        self.assertEqual(first.target, self.source.chunk_id)

        self.controller.begin_action(first)
        self.controller.observe(
            first,
            self.source.content,
            actual_cost=20,
        )

    def test_propagation_changes_next_action(self):
        baseline_next = self.controller.select_next().target

        self.assertEqual(
            baseline_next,
            self.unrelated.chunk_id,
        )

        records = rerank_pending_actions(
            controller=self.controller,
            tracker=self.tracker,
            chunks=self.chunks,
            source_location=self.source.chunk_id,
            support=1,
        )

        adaptive_next = self.controller.select_next().target

        self.assertEqual(
            adaptive_next,
            self.related.chunk_id,
        )

        self.assertEqual(len(records), 1)

        self.assertEqual(
            records[0].relationship,
            "same_method",
        )

    def test_negative_evidence_reduces_confidence(self):
        rerank_pending_actions(
            controller=self.controller,
            tracker=self.tracker,
            chunks=self.chunks,
            source_location=self.source.chunk_id,
            support=-1,
        )

        self.assertAlmostEqual(
            self.tracker.hypotheses[
                self.related.chunk_id
            ].confidence,
            0.2,
        )

    def test_neutral_evidence_does_not_rerank(self):
        before = self.controller.select_next().target

        records = rerank_pending_actions(
            controller=self.controller,
            tracker=self.tracker,
            chunks=self.chunks,
            source_location=self.source.chunk_id,
            support=0,
        )

        after = self.controller.select_next().target

        self.assertEqual(records, [])
        self.assertEqual(before, after)

    def test_completed_action_is_not_reintroduced(self):
        rerank_pending_actions(
            controller=self.controller,
            tracker=self.tracker,
            chunks=self.chunks,
            source_location=self.source.chunk_id,
            support=1,
        )

        pending_targets = {
            action.target
            for action in self.controller.state.pending_actions.values()
        }

        self.assertNotIn(
            self.source.chunk_id,
            pending_targets,
        )

    def test_budget_is_unchanged_by_reranking(self):
        before = self.controller.state.remaining_budget

        rerank_pending_actions(
            controller=self.controller,
            tracker=self.tracker,
            chunks=self.chunks,
            source_location=self.source.chunk_id,
            support=1,
        )

        self.assertEqual(
            self.controller.state.remaining_budget,
            before,
        )


if __name__ == "__main__":
    unittest.main()
