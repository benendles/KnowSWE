import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from knowswe.models.gemma_client import GemmaClient


class TestTruncatedPrompt(unittest.TestCase):

    @patch("knowswe.models.gemma_client.subprocess.run")
    def test_truncated_prompt(self, mock_run):
        mock_run.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "llama.cpp banner\n"
                "\n> Analyze this code ... (truncated)\n"
                '```json\n{"support": 0.5, "reason": "test"}\n```\n'
                "[ Prompt: 73.4 t/s | Generation: 23.1 t/s ]\n"
                "Exiting...\n"
            ),
            stderr="",
        )

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            executable = Path(directory) / "llama-cli"

            model.touch()
            executable.touch()

            client = GemmaClient(
                model_path=model,
                executable=executable,
            )

            result = client.generate_json(
                "A sufficiently long original prompt"
            )

        self.assertEqual(result["support"], 0.5)
        self.assertEqual(result["reason"], "test")


if __name__ == "__main__":
    unittest.main()
