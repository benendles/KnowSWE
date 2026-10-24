import argparse
import json
import subprocess
from pathlib import Path

from experiments.evaluate_chunks import evaluate as evaluate_e3
from knowswe.evaluation.benchmark import evaluate_dataset


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument(
        "--output",
        default="results/E4_Evaluation/multi_bug/results.json",
    )
    args = parser.parse_args()

    cases = json.loads(
        Path(args.manifest).read_text(encoding="utf-8")
    )

    repo = Path("data/real_repos/sqlfluff").resolve()
    original_commit = git(repo, "rev-parse", "HEAD")

    if git(repo, "status", "--porcelain"):
        raise RuntimeError(
            "SQLFluff checkout has local changes. "
            "Commit or preserve them before running."
        )

    results = []

    try:
        for index, case in enumerate(cases, 1):
            print(f"\n[{index}/{len(cases)}] {case['id']}", flush=True)

            git(repo, "checkout", "--detach", case["base_commit"])

            benchmark_case = {
                "id": case["id"],
                "repository": str(repo),
                "bug_report": case["bug_report"],
                "faulty_files": case["faulty_files"],
            }

            e3 = evaluate_e3(
                benchmark_case,
                budget=args.budget,
                chunk_size=args.chunk_size,
            )

            results.append({
                "id": case["id"],
                "base_commit": case["base_commit"],
                "reference_files": case["faulty_files"],
                "E3": e3,
            })

            print(
                f"E3 hit={e3['hit']} "
                f"tokens={e3['estimated_tokens_used']}",
                flush=True,
            )

    finally:
        git(repo, "checkout", "--detach", original_commit)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nKnowSWE — Multi-Bug Evaluation")
    print("=" * 65)

    hits = sum(result["E3"]["hit"] for result in results)

    print("Cases:", len(results))
    print("E3 hits:", hits)
    print("E3 hit rate:", f"{hits / len(results):.2%}")
    print("Results saved:", destination)


if __name__ == "__main__":
    main()
