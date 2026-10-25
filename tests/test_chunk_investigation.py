import tempfile
import unittest
from pathlib import Path

from knowswe.beds.chunk_runner import (
    investigate_repository_chunks,
)


class TestChunkInvestigation(unittest.TestCase):

    def test_chunk_investigation_respects_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (root / "main.py").write_text(
                "def process_invoice(amount):\n"
                "    return amount * 2\n"
                "\n"
                "def calculate_tax(amount):\n"
                "    return amount * 0.2\n"
            )

            state = investigate_repository_chunks(
                repository_path=directory,
                bug_report="process_invoice calculation",
                budget=100,
                top_k=1,
                max_steps=10,
                max_chunk_tokens=30,
            )

            self.assertLessEqual(state.spent_budget, 100)
            self.assertGreater(len(state.evidence), 0)

            inspected = [
                entry["target"]
                for entry in state.history
            ]

            self.assertEqual(
                len(inspected),
                len(set(inspected)),
            )

    def test_small_budget_limits_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (root / "main.py").write_text(
                "def process_invoice(amount):\n"
                "    return amount * 2\n"
                "\n"
                "def calculate_tax(amount):\n"
                "    return amount * 0.2\n"
            )

            state = investigate_repository_chunks(
                repository_path=directory,
                bug_report="process_invoice calculation",
                budget=15,
                top_k=1,
                max_steps=10,
                max_chunk_tokens=30,
            )

            self.assertLessEqual(state.spent_budget, 15)

    def test_chunk_targets_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            (root / "main.py").write_text(
                "def alpha():\n"
                "    return 1\n"
                "\n"
                "def beta():\n"
                "    return 2\n"
            )

            state = investigate_repository_chunks(
                repository_path=directory,
                bug_report="alpha beta",
                budget=100,
                top_k=1,
                max_steps=10,
                max_chunk_tokens=20,
            )

            targets = [
                entry["target"]
                for entry in state.history
            ]

            self.assertEqual(len(targets), len(set(targets)))


    def test_invalid_relevance_mode(self):
        from tempfile import TemporaryDirectory
        from knowswe.beds.chunk_environment import (
            ChunkInvestigationEnvironment,
        )

        with TemporaryDirectory() as directory:
            (Path(directory) / "example.py").write_text(
                "def example():\n    return 1\n"
            )

            with self.assertRaises(ValueError):
                ChunkInvestigationEnvironment(
                    directory,
                    relevance_mode="invalid",
                )

    def test_chunk_relevance_mode_runs(self):
        from tempfile import TemporaryDirectory
        from knowswe.beds.chunk_runner import (
            investigate_repository_chunks,
        )

        with TemporaryDirectory() as directory:
            (Path(directory) / "example.py").write_text(
                "def calculate_invoice_total(amount):\n"
                "    return amount * 1.2\n"
            )

            state = investigate_repository_chunks(
                repository_path=directory,
                bug_report="calculate invoice total",
                budget=100,
                top_k=1,
                max_steps=5,
                max_chunk_tokens=100,
                relevance_mode="chunk",
            )

            self.assertLessEqual(
                state.spent_budget,
                state.initial_budget,
            )


if __name__ == "__main__":
    unittest.main()
