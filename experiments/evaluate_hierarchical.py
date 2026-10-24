import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from knowswe.chunking.ast_chunker import chunk_python_file
from knowswe.retrieval.bm25 import RepositoryRetriever


def tokenize(text):
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", text.lower())


def evaluate(case, budget=2000, chunk_size=500, file_weight=0.4):
    root = Path(case["repository"]).resolve()

    file_retriever = RepositoryRetriever(str(root))
    file_retriever.index()

    file_results = file_retriever.search(
        case["bug_report"],
        top_k=len(file_retriever.files),
    )

    file_scores = {
        item["file"]: item["score"]
        for item in file_results
    }

    chunks = []

    for path in sorted(root.rglob("*.py")):
        if ".git" in path.parts or ".venv" in path.parts:
            continue

        chunks.extend(
            chunk_python_file(
                path,
                root,
                max_tokens=chunk_size,
            )
        )

    corpus = [
        tokenize(
            chunk.filename + " " +
            chunk.symbol + " " +
            chunk.content
        )
        for chunk in chunks
    ]

    chunk_bm25 = BM25Okapi(corpus)
    chunk_scores = chunk_bm25.get_scores(
        tokenize(case["bug_report"])
    )

    max_chunk_score = max(chunk_scores) or 1
    max_file_score = max(file_scores.values()) or 1

    combined = []

    for index, chunk in enumerate(chunks):
        chunk_score = chunk_scores[index] / max_chunk_score
        file_score = (
            file_scores.get(chunk.filename, 0)
            / max_file_score
        )

        score = (
            (1 - file_weight) * chunk_score
            + file_weight * file_score
        )

        combined.append((score, index))

    combined.sort(key=lambda item: (-item[0], item[1]))

    selected = []
    remaining = budget

    for score, index in combined:
        chunk = chunks[index]

        if chunk.estimated_tokens <= remaining:
            selected.append({
                "file": chunk.filename,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "estimated_tokens": chunk.estimated_tokens,
                "score": float(score),
            })

            remaining -= chunk.estimated_tokens

        if remaining == 0:
            break

    reference_files = set(case["faulty_files"])

    selected_files = {
        item["file"]
        for item in selected
    }

    hits = selected_files & reference_files

    return {
        "case_id": case["id"],
        "budget": budget,
        "file_weight": file_weight,
        "hit": bool(hits),
        "recall": len(hits) / len(reference_files),
        "estimated_tokens_used": budget - remaining,
        "selected_chunks": selected,
        "reference_files": sorted(reference_files),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--file-weight", type=float, default=0.4)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    cases = json.loads(
        Path(args.dataset).read_text()
    )

    results = [
        evaluate(
            case,
            budget=args.budget,
            chunk_size=args.chunk_size,
            file_weight=args.file_weight,
        )
        for case in cases
    ]

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    for result in results:
        print(
            result["case_id"],
            "hit=", result["hit"],
            "tokens=", result["estimated_tokens_used"],
        )


if __name__ == "__main__":
    main()
