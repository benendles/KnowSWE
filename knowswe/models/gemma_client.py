"""Local Gemma inference through llama.cpp."""

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = Path(
    "/opt/knowswe/models/gemma3-1b/"
    "gemma-3-1b-it-Q4_K_M.gguf"
)

DEFAULT_EXECUTABLE = Path(
    "/opt/knowswe/llama.cpp/build/bin/llama-cli"
)


@dataclass
class GemmaResponse:
    text: str
    elapsed_seconds: float
    raw_output: str


class GemmaClient:
    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        executable=DEFAULT_EXECUTABLE,
        timeout=120,
        context_size=2048,
        max_tokens=256,
        threads=2,
    ):
        self.model_path = Path(model_path)
        self.executable = Path(executable)
        self.timeout = timeout
        self.context_size = context_size
        self.max_tokens = max_tokens
        self.threads = threads

        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)

        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)

    def generate(self, prompt: str) -> GemmaResponse:
        command = [
            str(self.executable),
            "--model", str(self.model_path),
            "--prompt", prompt,
            "--single-turn",
            "--simple-io",
            "--no-display-prompt",
            "--temp", "0",
            "--ctx-size", str(self.context_size),
            "--predict", str(self.max_tokens),
            "--threads", str(self.threads),
        ]

        started = time.monotonic()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Gemma inference exceeded {self.timeout}s"
            ) from exc

        elapsed = time.monotonic() - started

        raw = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            raise RuntimeError(
                f"Gemma process failed: {raw[-1000:]}"
            )

        if "Failed to initialize samplers" in raw:
            raise RuntimeError(
                "Gemma sampler initialization failed"
            )

        # llama-cli simple-io may include the prompt and
        # performance statistics in its output.
        # llama-cli displays either the full prompt or a
        # shortened preview ending with "... (truncated)".
        marker = "> " + prompt

        if marker in raw:
            response_start = raw.index(marker) + len(marker)
            response_text = raw[response_start:]
        else:
            prompt_marker = raw.find("\n> ")

            if prompt_marker >= 0:
                preview_start = prompt_marker + len("\n> ")
            elif raw.startswith("> "):
                preview_start = 2
            else:
                raise RuntimeError(
                    "Could not locate Gemma prompt marker"
                )
            truncation_marker = "... (truncated)"

            truncation_end = raw.find(
                truncation_marker,
                preview_start,
            )

            if truncation_end == -1:
                raise RuntimeError(
                    "Could not locate truncated prompt boundary"
                )

            response_text = raw[
                truncation_end + len(truncation_marker):
            ]

        # Remove llama.cpp performance statistics and exit text.
        response_text = re.split(
            r"(?m)^\[ Prompt:",
            response_text,
            maxsplit=1,
        )[0].strip()

        if response_text.endswith("Exiting..."):
            response_text = response_text[
                :-len("Exiting...")
            ].strip()

        if not response_text:
            raise RuntimeError(
                "Gemma returned an empty response"
            )

        return GemmaResponse(
            text=response_text,
            elapsed_seconds=elapsed,
            raw_output=raw,
        )

    def generate_json(self, prompt: str) -> dict:
        response = self.generate(prompt)

        text = response.text.strip()

        # Accept a single fenced JSON object.
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if fenced:
            text = fenced.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemma returned invalid JSON: {text[:500]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                "Gemma response must be a JSON object"
            )

        return parsed
