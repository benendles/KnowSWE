import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/G5_Baseline_Ablation"

MODEL = Path(
    "/opt/knowswe/models/gemma3-1b/"
    "gemma-3-1b-it-Q4_K_M.gguf"
)

LLAMA = Path(
    "/opt/knowswe/llama.cpp/build/bin/llama-cli"
)

BASE_PROMPT = (
    ROOT / "results/G3_Gemma_Real_Bugs/context_8192/"
    "sqlfluff_1625_prompt.txt"
)

BM25_RESULT = (
    ROOT / "results/G3_Gemma_Real_Bugs/context_8192/"
    "sqlfluff_1625_g3.json"
)

CASE_ID = "sqlfluff__sqlfluff-1625"

ANSWER_MARKER = (
    "Return your answer using exactly this format:"
)


def split_prompt(prompt):
    pattern = re.compile(r"(?m)^Evidence \d+\s*$")
    matches = list(pattern.finditer(prompt))

    assert len(matches) == 14

    suffix_start = prompt.rfind(ANSWER_MARKER)
    assert suffix_start != -1

    prefix = prompt[:matches[0].start()]
    suffix = prompt[suffix_start:]

    blocks = []

    for i, match in enumerate(matches):
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else suffix_start
        )

        block = prompt[match.start():end]

        file_match = re.search(
            r"(?m)^File:\s*(.+)$", block
        )

        assert file_match is not None

        blocks.append({
            "rank": i + 1,
            "file": file_match.group(1).strip(),
            "text": block,
        })

    return prefix, blocks, suffix


def parse_prediction(raw):
    raw = raw.replace("\r\n", "\n")

    marker = "\nAssistant:\n"

    if marker not in raw:
        raise ValueError("Assistant marker not found")

    answer = raw.rsplit(marker, 1)[1].strip()

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


def run_model(directory, prompt):
    directory.mkdir(parents=True, exist_ok=True)

    prompt_path = directory / "prompt.txt"
    output_path = directory / "model_output.txt"
    runtime_path = directory / "runtime_stdout.txt"

    prompt_path.write_text(prompt)

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

    with runtime_path.open("w") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Inference failed: {completed.returncode}"
        )

    prediction = parse_prediction(
        output_path.read_text(errors="replace")
    )

    result = {
        "case_id": CASE_ID,
        "model": MODEL.name,
        "context_size": 8192,
        "temperature": 0,
        "seed": 42,
        "max_generation_tokens": 256,
        "prompt_sha256": hashlib.sha256(
            prompt.encode()
        ).hexdigest(),
        "prediction": prediction,
    }

    (directory / "result.json").write_text(
        json.dumps(result, indent=2)
    )

    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    original = BASE_PROMPT.read_text()

    prefix, blocks, suffix = split_prompt(original)

    # A: No repository evidence.
    prompt_a = prefix + suffix

    # B: Original BM25 evidence.
    prompt_b = (
        prefix
        + "".join(b["text"] for b in blocks)
        + suffix
    )

    assert prompt_b == original

    # C: Oracle evidence.
    # Only the implementation chunks for the known
    # faulty file. Not a deployable retrieval strategy.
    oracle_blocks = [
        b for b in blocks
        if b["file"] == "src/sqlfluff/rules/L031.py"
    ]

    assert len(oracle_blocks) == 2

    prompt_c = (
        prefix
        + "".join(b["text"] for b in oracle_blocks)
        + suffix
    )

    configurations = {
        "G5_A_no_retrieval": prompt_a,
        "G5_C_oracle_implementation": prompt_c,
    }

    for name, prompt in configurations.items():
        print(f"\n=== RUNNING {name} ===", flush=True)

        result = run_model(
            OUTPUT / name,
            prompt,
        )

        print(
            json.dumps(
                result["prediction"],
                indent=2,
            ),
            flush=True,
        )

    # Reuse existing B result without new inference.
    baseline = json.loads(BM25_RESULT.read_text())

    directory = OUTPUT / "G5_B_BM25"
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "result.json").write_text(
        json.dumps(baseline, indent=2)
    )

    print("\nG5 completed.", flush=True)


if __name__ == "__main__":
    main()
