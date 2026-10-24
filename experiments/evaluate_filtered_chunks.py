import json
import argparse
from pathlib import Path

from rank_bm25 import BM25Okapi

from experiments.evaluate_chunks import (
    build_chunk_index,
    tokenize,
)

from knowswe.beds.chunk_controller import (
    ChunkCandidate,
    ChunkBEDSController,
)


def evaluate(case, budget=2000, chunk_size=500):
    root = Path(case["repository"]).resolve()

    all_chunks = build_chunk_index(root, chunk_size)

    # Exclude whitespace-only chunks.
    chunks = [
        chunk
        for chunk in all_chunks
        if chunk.content.strip()
    ]

    corpus = [
        tokenize(
            chunk.filename + " " +
            chunk.symbol + " " +
            chunk.content
        )
        for chunk in chunks
    ]

    bm25 = BM25Okapi(corpus)

    scores = bm25.get_scores(
        tokenize(case["bug_report"])
    )

    ranking = sorted(
        range(len(chunks)),
        key=lambda i: (-scores[i], i),
    )

    # E6a: filtered BM25 + greedy budget.
    selected_a = []
    remaining_a = budget

    for index in ranking:
        chunk = chunks[index]

        if chunk.estimated_tokens <= remaining_a:
            selected_a.append(index)
            remaining_a -= chunk.estimated_tokens

        if remaining_a == 0:
            break

    # E6b: same filtered chunks + BEDS selection.
    max_score = max(
        (float(score) for score in scores),
        default=0.0,
    )

    if max_score <= 0:
        max_score = 1.0

    candidates = [
        ChunkCandidate(
            index=i,
            filename=chunk.filename,
            relevance=max(0.0, float(scores[i])) / max_score,
            cost=chunk.estimated_tokens,
        )
        for i, chunk in enumerate(chunks)
    ]

    controller = ChunkBEDSController(budget=budget)

    selected_b = []

    while candidates:
        candidate = controller.select(candidates)

        if candidate is None:
            break

        selected_b.append(candidate.index)
        candidates.remove(candidate)

    def summarize(selected_indices):
        selected = []

        for index in selected_indices:
            chunk = chunks[index]

            selected.append({
                "file": chunk.filename,
                "symbol": chunk.symbol,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "estimated_tokens": chunk.estimated_tokens,
            })

        reference = set(case["faulty_files"])
        files = {chunk["file"] for chunk in selected}

        return {
            "hit": bool(files & reference),
            "selected_chunks": selected,
            "estimated_tokens_used": sum(
                chunk["estimated_tokens"]
                for chunk in selected
            ),
        }

    return {
        "id": case["id"],
        "budget": budget,
        "original_chunk_count": len(all_chunks),
        "filtered_chunk_count": len(chunks),
        "E6a": summarize(selected_a),
        "E6b": summarize(selected_b),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()

    cases = json.loads(
        Path(args.dataset).read_text()
    )

    results = [
        evaluate(case, args.budget, args.chunk_size)
        for case in cases
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    for row in results:
        print(
            row["id"],
            "E6a=", row["E6a"]["hit"],
            "E6b=", row["E6b"]["hit"],
            "indexed=", row["filtered_chunk_count"],
        )

    print("\nResults saved:", output)


if __name__ == "__main__":
    main()
