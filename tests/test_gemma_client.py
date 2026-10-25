import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from subprocess import CompletedProcess

from knowswe.models.gemma_client import GemmaClient


class TestGemmaClient(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()

        self.model = Path(self.directory.name) / "model.gguf"
        self.executable = Path(self.directory.name) / "llama-cli"

        self.model.touch()
        self.executable.touch()

        self.client = GemmaClient(
            model_path=self.model,
            executable=self.executable,
        )

    def tearDown(self):
        self.directory.cleanup()

    @patch("knowswe.models.gemma_client.subprocess.run")
    def test_successful_json(self, mock_run):
        mock_run.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '> Test prompt\n'
                '{"support": 0.5, "reason": "test"}\n'
                '[ Prompt: 50.0 t/s | Generation: 20.0 t/s ]\n'
                'Exiting...\n'
            ),
            stderr="",
        )

        result = self.client.generate_json("Test prompt")

        self.assertEqual(result["support"], 0.5)

    @patch("knowswe.models.gemma_client.subprocess.run")
    def test_sampler_failure(self, mock_run):
        mock_run.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "> Test prompt\n"
                "Error: Failed to initialize samplers: "
                "std::exception\n"
                "Exiting...\n"
            ),
            stderr="",
        )

        with self.assertRaises(RuntimeError):
            self.client.generate("Test prompt")

    @patch("knowswe.models.gemma_client.subprocess.run")
    def test_invalid_json(self, mock_run):
        mock_run.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "> Test prompt\n"
                "This is not JSON\n"
                "[ Prompt: 50.0 t/s | Generation: 20.0 t/s ]\n"
                "Exiting...\n"
            ),
            stderr="",
        )

        with self.assertRaises(ValueError):
            self.client.generate_json("Test prompt")


if __name__ == "__main__":
    unittest.main()
