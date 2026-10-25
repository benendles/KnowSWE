import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_PROMPT = (
    ROOT
    / "results/G3_Gemma_Real_Bugs/context_8192"
    / "sqlfluff_1625_prompt.txt"
)

BASELINE_RESULT = (
    ROOT
    / "results/G3_Gemma_Real_Bugs/context_8192"
    / "sqlfluff_1625_g3.json"
)

OUTPUT = ROOT / "results/G4_Evidence_Ordering"

MODEL = Path(
    "/opt/knowswe/models/gemma3-1b/"
    "gemma-3-1b-it-Q4_K_M.gguf"
)

LLAMA = Path(
    "/opt/knowswe/llama.cpp/build/bin/llama-cli"
)

CASE_ID = "sqlfluff__sqlfluff-1625"

SETTINGS = {
    "context_size": 8192,
    "max_generation_tokens": 256,
    "temperature": 0,
    "seed": 42,
    "threads": 2,
}


def sha256(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def split_prompt(prompt):
    pattern = re.compile(r"(?m)^Evidence \d+\s*$")
    matches = list(pattern.finditer(prompt))

    if len(matches) != 14:
        raise ValueError(
            f"Expected 14 evidence blocks, found {len(matches)}"
        )

    prefix = prompt[:matches[0].start()]

    suffix_marker = "Return your answer using exactly this format:"

    suffix_start = prompt.rfind(suffix_marker)

    if suffix_start == -1:
        raise ValueError("Answer-format instructions not found")

    suffix = prompt[suffix_start:]

    blocks = []

    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else suffix_start
        )

        block = prompt[match.start():end]

        file_match = re.search(
            r"(?m)^File:\s*(.+)$",
            block
        )

        if not file_match:
            raise ValueError("Evidence block missing filename")

        blocks.append({
            "original_rank": index + 1,
            "file": file_match.group(1).strip(),
            "text": block,
        })

    return prefix, blocks, suffix


def build_prompt(prefix, blocks, suffix):
    return prefix + "".join(
        block["text"] for block in blocks
    ) + suffix


def parse_answer(raw):
    normalized = raw.replace("\r\n", "\n")

    marker = "\nAssistant:\n"

    if marker not in normalized:
        raise ValueError("Assistant response marker not found")

    answer = normalized.rsplit(marker, 1)[1].strip()

    file_match = re.search(
        r"^FAULTY_FILE:\s*(.+)$",
        answer,
        re.MULTILINE
    )

    symbol_match = re.search(
        r"^FAULTY_SYMBOL:\s*(.+)$",
        answer,
        re.MULTILINE
    )

    return {
        "predicted_file": (
            file_match.group(1).strip()
            if file_match else None
        ),
        "predicted_symbol": (
            symbol_match.group(1).strip()
            if symbol_match else None
        ),
        "generated_answer": answer,
    }


def run_model(prompt_path, output_path, log_path):
    command = [
        str(LLAMA),
        "-m", str(MODEL),
        "-t", "2",
        "-c", "8192",
        "-n", "256",
        "--temp", "0",
        "--seed", "42",
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "--output-file", str(output_path),
        "-f", str(prompt_path),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
    )

    log_path.write_text(
        completed.stdout + "\n" + completed.stderr
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"llama.cpp failed: {completed.returncode}"
        )

    if not output_path.exists():
        raise FileNotFoundError(output_path)

    return parse_answer(
        output_path.read_text(errors="replace")
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    original_prompt = SOURCE_PROMPT.read_text()

    prefix, blocks, suffix = split_prompt(original_prompt)

    configurations = {
        "G4_A_original": blocks,
        "G4_B_implementation_first": sorted(
            blocks,
            key=lambda b: (
                0 if b["file"].startswith("src/") else 1,
                b["original_rank"],
            ),
        ),
        "G4_C_reversed": list(reversed(blocks)),
    }

    baseline = json.loads(BASELINE_RESULT.read_text())

    for name, ordered_blocks in configurations.items():
        directory = OUTPUT / name
        directory.mkdir(parents=True, exist_ok=True)

        prompt = build_prompt(prefix, ordered_blocks, suffix)

        prompt_path = directory / "prompt.txt"
        prompt_path.write_text(prompt)

        result_path = directory / "result.json"

        if name == "G4_A_original":
            if prompt != original_prompt:
                raise ValueError(
                    "G4-A prompt differs from original baseline"
                )

            prediction = baseline["prediction"]

        else:
            prediction = run_model(
                prompt_path,
                directory / "model_output.txt",
                directory / "runtime_stdout.txt",
            )

        result = {
            "experiment": name,
            "case_id": CASE_ID,
            "model": MODEL.name,
            "settings": SETTINGS,
            "prompt_sha256": sha256(prompt),
            "evidence_order": [
                {
                    "original_rank": block["original_rank"],
                    "file": block["file"],
                }
                for block in ordered_blocks
            ],
            "prediction": prediction,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        result_path.write_text(
            json.dumps(result, indent=2)
        )

        print()
        print("===", name, "===")
        print("Prompt SHA256:", result["prompt_sha256"])
        print("Predicted file:", prediction["predicted_file"])
        print("Predicted symbol:", prediction["predicted_symbol"])
        print("Answer:")
        print(prediction["generated_answer"])

    print("\nG4 experiment completed.")


if __name__ == "__main__":
    main()
