import json
import re
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CASE_ID = "sqlfluff__sqlfluff-1625"

REPO = ROOT / "data/real_repos/sqlfluff"

MODEL = Path(
    "/opt/knowswe/models/gemma3-1b/"
    "gemma-3-1b-it-Q4_K_M.gguf"
)

LLAMA = Path(
    "/opt/knowswe/llama.cpp/build/bin/llama-cli"
)

OUTPUT = ROOT / "results/G3_Gemma_Real_Bugs"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_case():
    dataset = json.loads(
        (
            ROOT /
            "data/benchmarks/swebench_lite/dev_labeled.json"
        ).read_text()
    )

    case = next(
        x for x in dataset
        if x["id"] == CASE_ID
    )

    # Strict separation: only allow the bug report and commit.
    return {
        "id": case["id"],
        "bug_report": case["bug_report"],
        "base_commit": case["base_commit"],
    }


def load_evidence():
    records = json.loads(
        (
            ROOT /
            "results/E4_Evaluation/multi_bug/"
            "sqlfluff_five_E3.json"
        ).read_text()
    )

    record = next(
        x for x in records
        if x["id"] == CASE_ID
    )

    return record["E3"]["selected_chunks"]


def historical_file(commit, filename):
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "show",
            f"{commit}:{filename}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout


def reconstruct_evidence(commit, chunks):
    evidence = []
    cache = {}

    for index, chunk in enumerate(chunks, start=1):
        filename = chunk["file"]

        if filename not in cache:
            cache[filename] = historical_file(
                commit,
                filename
            ).splitlines()

        lines = cache[filename]

        start = chunk["start_line"]
        end = chunk["end_line"]

        if not (1 <= start <= end <= len(lines)):
            raise ValueError(
                f"Invalid chunk range: {filename}:{start}-{end}"
            )

        code = "\n".join(
            f"{line_number}: {lines[line_number - 1]}"
            for line_number in range(start, end + 1)
        )

        evidence.append({
            "rank": index,
            "file": filename,
            "symbol": chunk["symbol"],
            "start_line": start,
            "end_line": end,
            "estimated_tokens": chunk["estimated_tokens"],
            "code": code,
        })

    return evidence


def build_prompt(case, evidence):
    sections = []

    for chunk in evidence:
        sections.append(
            f"Evidence {chunk['rank']}\n"
            f"File: {chunk['file']}\n"
            f"Symbol: {chunk['symbol']}\n"
            f"Lines: {chunk['start_line']}-"
            f"{chunk['end_line']}\n"
            f"```python\n{chunk['code']}\n```"
        )

    evidence_text = "\n\n".join(sections)

    return f"""You are a Python repository debugging assistant.

Your task is fault localization, not code repair.

Read the bug report and the retrieved repository evidence.

Identify the most likely faulty source file and function or class.

Do not invent source files that are not in the evidence.

If the evidence is insufficient, say so.

BUG REPORT:
{case['bug_report']}

RETRIEVED SOURCE EVIDENCE:
{evidence_text}

Return your answer using exactly this format:

FAULTY_FILE: <repository-relative path>
FAULTY_SYMBOL: <function or class name>
REASON: <brief explanation>
"""


def run_gemma(prompt):
    output_file = OUTPUT / "sqlfluff_1625_model_output.txt"

    if output_file.exists():
        output_file.unlink()

    command = [
        str(LLAMA),
        "-m", str(MODEL),
        "-t", "2",
        "-c", "4096",
        "-n", "256",
        "--temp", "0",
        "--seed", "42",
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "--output-file", str(output_file),
        "-p", prompt,
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
    )

    model_output = (
        output_file.read_text(errors="replace")
        if output_file.exists()
        else ""
    )

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "model_output": model_output,
        "output_file": str(output_file),
    }


def parse_prediction(raw):
    """Extract only Gemma's generated response."""

    # Normalize line endings.
    raw = raw.replace("\r\n", "\n")

    # llama.cpp output contains User: and Assistant: sections.
    parts = raw.rsplit("\nAssistant:\n", 1)

    if len(parts) != 2:
        raise ValueError(
            "Assistant response marker not found."
        )

    answer = parts[1].strip()

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


def main():
    assert MODEL.exists(), f"Model missing: {MODEL}"
    assert LLAMA.exists(), f"llama-cli missing: {LLAMA}"

    case = load_case()
    chunks = load_evidence()

    evidence = reconstruct_evidence(
        case["base_commit"],
        chunks
    )

    prompt = build_prompt(case, evidence)

    prompt_path = OUTPUT / "sqlfluff_1625_prompt.txt"
    prompt_path.write_text(prompt)

    print("=== G3 GEMMA REAL BUG EXPERIMENT ===")
    print("Case:", CASE_ID)
    print("Commit:", case["base_commit"])
    print("Retrieved chunks:", len(evidence))
    print("Prompt characters:", len(prompt))
    print("Prompt saved:", prompt_path)

    print("\nRunning Gemma 3 1B...\n", flush=True)

    inference = run_gemma(prompt)

    prediction = parse_prediction(
        inference["model_output"]
    )

    result = {
        "case_id": CASE_ID,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "base_commit": case["base_commit"],
        "model": MODEL.name,
        "model_sha256": hashlib.sha256(
            MODEL.read_bytes()
        ).hexdigest(),
        "retrieval": "E3_AST_chunk_BM25",
        "retrieved_chunks": len(evidence),
        "evidence_budget_estimated": 2000,
        "context_size": 4096,
        "temperature": 0,
        "seed": 42,
        "prompt_sha256": hashlib.sha256(
            prompt.encode()
        ).hexdigest(),
        "prediction": prediction,
        "inference": inference,
    }

    output_path = OUTPUT / "sqlfluff_1625_g3.json"

    output_path.write_text(
        json.dumps(result, indent=2)
    )

    print("=== GEMMA PREDICTION ===")
    print(json.dumps(prediction, indent=2))

    print("\nReturn code:", inference["returncode"])
    print("Results saved:", output_path)

    if inference["returncode"] != 0:
        print("\nInference failed. Inspect saved stderr.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
