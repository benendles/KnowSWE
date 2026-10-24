import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from knowswe.chunking.ast_chunker import chunk_python_file


def tokenize(text):
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", text.lower())


def build_chunk_index(repository, max_tokens):
    chunks = []

    for path in sorted(repository.rglob("*.py")):
        if ".git" in path.parts or ".venv" in path.parts:
            continue

        chunks.extend(
            chunk_python_file(
                path,
                repository,
                max_tokens=max_tokens,
            )
        )

    return chunks


def evaluate(case, budget, chunk_size):
    repository = Path(case["repository"]).resolve()

    chunks = build_chunk_index(repository, chunk_size)

    corpus = [
        tokenize(
            chunk.filename + " " +
            chunk.symbol + " " +
            chunk.content
        )
        for chunk in chunks
    ]

    bm25 = BM25Okapi(corpus)

    query = tokenize(case["bug_report"])
    scores = bm25.get_scores(query)

    ranking = sorted(
        range(len(chunks)),
        key=lambda i: (-scores[i], i),
    )

    selected = []
    remaining = budget

    for index in ranking:
        chunk = chunks[index]

        if chunk.estimated_tokens <= remaining:
            selected.append({
                "file": chunk.filename,
                "symbol": chunk.symbol,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "estimated_tokens": chunk.estimated_tokens,
                "score": float(scores[index]),
            })

            remaining -= chunk.estimated_tokens

        if remaining == 0:
            break

    reference_files = set(case["faulty_files"])

    selected_files = {
        chunk["file"] for chunk in selected
    }

    hits = selected_files & reference_files

    return {
        "case_id": case["id"],
        "budget": budget,
        "chunk_size": chunk_size,
        "total_indexed_chunks": len(chunks),
        "selected_chunks": selected,
        "selected_files": sorted(selected_files),
        "hit": bool(hits),
        "recall": len(hits) / len(reference_files),
        "estimated_tokens_used": budget - remaining,
        "reference_files": sorted(reference_files),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    cases = json.loads(
        Path(args.dataset).read_text(encoding="utf-8")
    )

    results = [
        evaluate(case, args.budget, args.chunk_size)
        for case in cases
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("KnowSWE — E3 Chunk-Level BM25 Evaluation")
    print("=" * 65)

    for result in results:
        print("Case:", result["case_id"])
        print("Budget:", result["budget"])
        print("Chunk size:", result["chunk_size"])
        print("Indexed chunks:", result["total_indexed_chunks"])
        print("Selected chunks:", len(result["selected_chunks"]))
        print("Estimated tokens used:", result["estimated_tokens_used"])
        print("Reference fix-file hit:", result["hit"])
        print("Recall:", result["recall"])

        print("\nTop selected chunks:")

        for chunk in result["selected_chunks"][:10]:
            print(
                f"  {chunk['file']}:"
                f"{chunk['start_line']}-{chunk['end_line']}"
                f" ({chunk['estimated_tokens']} tokens)"
            )

    print("\nResults saved to:", output)


if __name__ == "__main__":
    main()
