import unittest
from unittest.mock import patch
from types import SimpleNamespace

from knowswe.beds.adaptive_actions import (
    ActionType,
    InvestigationAction,
)
from knowswe.beds.resilient_gemma_runner import investigate_resiliently
from knowswe.chunking.method_chunker import MethodChunk
from knowswe.models.minimal_evidence_evaluator import MinimalEvidenceAssessment
from knowswe.models.resilient_evaluator import ResilientEvidenceEvaluator


def make_chunk(filename, start, symbol):
    return MethodChunk(
        filename=filename,
        start_line=start,
        end_line=start + 2,
        symbol=symbol,
        parent_symbol="Worker" if filename == "worker.py" else None,
        content="def example():\n    return 1\n",
        estimated_tokens=20,
    )


class FakeEnvironment:
    def __init__(self, *args, **kwargs):
        self.source = make_chunk("worker.py", 10, "Worker.process")
        self.related = make_chunk("worker.py", 20, "Worker.process")
        self.unrelated = make_chunk("other.py", 30, "Other.run")

        self.chunks = {
            chunk.chunk_id: chunk
            for chunk in (
                self.source,
                self.related,
                self.unrelated,
            )
        }

    def initial_actions(self, bug_report, top_k=5):
        return [
            InvestigationAction(
                action_type=ActionType.INSPECT,
                target=chunk.chunk_id,
                estimated_cost=20,
                relevance=relevance,
                diagnostic_value=0.5,
            )
            for chunk, relevance in [
                (self.source, 0.9),
                (self.related, 0.4),
                (self.unrelated, 0.5),
            ]
        ]

    def execute(self, action):
        chunk = self.chunks[action.target]

        return SimpleNamespace(
            content=chunk.content,
            chunk=chunk,
            discovered_actions=[],
        )


class FakeEvaluator:
    def assess(self, bug_report, location, content, start_line):
        return MinimalEvidenceAssessment(
            location=location,
            support=1,
            evidence_line=start_line,
            evidence_text=content.splitlines()[0],
        )


class TestPropagationIntegration(unittest.TestCase):

    def run_investigation(self, enabled):
        evaluator = ResilientEvidenceEvaluator(
            FakeEvaluator(),
            max_retries=0,
        )

        with patch(
            "knowswe.beds.resilient_gemma_runner."
            "ChunkInvestigationEnvironment",
            FakeEnvironment,
        ):
            return investigate_resiliently(
                repository_path="unused",
                bug_report="worker process bug",
                evaluator=evaluator,
                budget=100,
                max_steps=2,
                enable_propagation=enabled,
            )

    def test_propagation_changes_second_inspection(self):
        baseline = self.run_investigation(False)
        adaptive = self.run_investigation(True)

        self.assertEqual(
            baseline.steps[0].location,
            adaptive.steps[0].location,
        )

        self.assertEqual(
            baseline.steps[1].location,
            "other.py:30:32",
        )

        self.assertEqual(
            adaptive.steps[1].location,
            "worker.py:20:22",
        )

    def test_both_modes_respect_budget(self):
        for enabled in (False, True):
            result = self.run_investigation(enabled)

            self.assertEqual(len(result.steps), 2)
            self.assertEqual(result.state.spent_budget, 40)
            self.assertEqual(len(result.state.history), 2)


if __name__ == "__main__":
    unittest.main()
