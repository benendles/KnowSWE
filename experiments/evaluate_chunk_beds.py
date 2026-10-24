import argparse
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from experiments.evaluate_chunks import (
    tokenize,
    build_chunk_index,
)

from knowswe.beds.chunk_controller import (
    ChunkCandidate,
    ChunkBEDSController,
)


def evaluate(case, budget=2000, chunk_size=500):
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

    scores = bm25.get_scores(
        tokenize(case["bug_report"])
    )

    max_score = max(
        (float(score) for score in scores),
        default=0.0,
    )

    if max_score <= 0:
        max_score = 1.0

    candidates = [
        ChunkCandidate(
            index=index,
            filename=chunk.filename,
            relevance=max(0.0, float(scores[index])) / max_score,
            cost=chunk.estimated_tokens,
        )
        for index, chunk in enumerate(chunks)
    ]

    controller = ChunkBEDSController(budget=budget)

    selected = []

    while candidates:
        candidate = controller.select(candidates)

        if candidate is None:
            break

        candidates.remove(candidate)

        chunk = chunks[candidate.index]

        selected.append({
            "file": chunk.filename,
            "symbol": chunk.symbol,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "estimated_tokens": chunk.estimated_tokens,
            "bm25_relevance": candidate.relevance,
        })

    reference_files = set(case["faulty_files"])

    selected_files = {
        chunk["file"]
        for chunk in selected
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
        "recall": (
            len(hits) / len(reference_files)
            if reference_files else 0.0
        ),
        "estimated_tokens_used": (
            budget - controller.remaining
        ),
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
        Path(args.dataset).read_text()
    )

    results = [
        evaluate(case, args.budget, args.chunk_size)
        for case in cases
    ]

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("KnowSWE — E5 Chunk-Level BEDS")
    print("=" * 65)

    for result in results:
        print(
            result["case_id"],
            "hit=", result["hit"],
            "tokens=", result["estimated_tokens_used"],
            "chunks=", len(result["selected_chunks"]),
        )

    print("\nResults saved:", destination)


if __name__ == "__main__":
    main()
