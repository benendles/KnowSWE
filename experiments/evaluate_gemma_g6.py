import json
import hashlib
from pathlib import Path

from experiments.evaluate_gemma_g5_ablation import (
    run_model,
    split_prompt,
)

ROOT = Path(__file__).resolve().parents[1]

CASE_ID = "sqlfluff__sqlfluff-1625"

OUTPUT = ROOT / "results/G6_BM25_vs_BEDS"

REPOSITORY = ROOT / "data/real_repos/sqlfluff"

BASE_PROMPT = (
    ROOT
    / "results/G3_Gemma_Real_Bugs/context_8192/"
    / "sqlfluff_1625_prompt.txt"
)

E5_RESULTS = (
    ROOT
    / "results/E4_Evaluation/multi_bug/"
    / "sqlfluff_five_E3_vs_E5.json"
)


def load_e5_chunks():
    data = json.loads(E5_RESULTS.read_text())

    case = next(
        row for row in data
        if row["id"] == CASE_ID
    )

    return case["E5"]["selected_chunks"]


def reconstruct_evidence(chunks):
    sections = []

    for rank, chunk in enumerate(chunks, start=1):
        filename = chunk["file"]

        path = REPOSITORY / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        lines = path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()

        start = chunk["start_line"]
        end = chunk["end_line"]

        if start < 1 or end < start or end > len(lines):
            raise ValueError(
                f"Invalid lines for {filename}: {start}-{end}"
            )

        code = "\n".join(lines[start - 1:end])

        sections.append(
            f"Evidence {rank}\n"
            f"File: {filename}\n"
            f"Symbol: {chunk['symbol']}\n"
            f"Lines: {start}-{end}\n"
            f"```python\n{code}\n```\n\n"
        )

    return "".join(sections)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    original = BASE_PROMPT.read_text()

    prefix, blocks, suffix = split_prompt(original)

    e5_chunks = load_e5_chunks()

    e5_evidence = reconstruct_evidence(e5_chunks)

    beds_prompt = prefix + e5_evidence + suffix

    print("=== G6: BM25 VS BEDS ===", flush=True)
    print("Case:", CASE_ID, flush=True)

    print(
        "BM25 chunks:",
        len(blocks),
        flush=True
    )

    print(
        "BEDS chunks:",
        len(e5_chunks),
        flush=True
    )

    print(
        "BM25 prompt characters:",
        len(original),
        flush=True
    )

    print(
        "BEDS prompt characters:",
        len(beds_prompt),
        flush=True
    )

    print(
        "BEDS prompt SHA256:",
        hashlib.sha256(
            beds_prompt.encode()
        ).hexdigest(),
        flush=True
    )

    print("\n=== RUNNING GEMMA WITH BEDS ===", flush=True)

    result = run_model(
        OUTPUT / "G6_B_BEDS",
        beds_prompt
    )

    print("\n=== BEDS PREDICTION ===")
    print(
        json.dumps(
            result["prediction"],
            indent=2
        )
    )

    print("\nG6 BEDS inference completed.")


if __name__ == "__main__":
    main()
