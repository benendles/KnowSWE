import json
import subprocess
from pathlib import Path

from experiments.evaluate_filtered_chunks import evaluate


ROOT = Path("data/real_repos/sqlfluff").resolve()

MANIFEST = Path(
    "data/benchmarks/swebench_lite/multi_bug/"
    "sqlfluff_five_cases.json"
)

OUTPUT = Path(
    "results/E4_Evaluation/multi_bug/"
    "sqlfluff_five_E6_filtered.json"
)


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
    ).strip()


def main():
    cases = json.loads(MANIFEST.read_text())

    if git("status", "--porcelain"):
        raise RuntimeError("SQLFluff checkout has local changes.")

    original_commit = git("rev-parse", "HEAD")
    results = []

    try:
        for i, case in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}] {case['id']}", flush=True)

            git("checkout", "--detach", case["base_commit"])

            benchmark_case = {
                "id": case["id"],
                "repository": str(ROOT),
                "bug_report": case["bug_report"],
                "faulty_files": case["faulty_files"],
            }

            result = evaluate(
                benchmark_case,
                budget=2000,
                chunk_size=500,
            )

            results.append(result)

            print(
                "E6a=", result["E6a"]["hit"],
                "E6b=", result["E6b"]["hit"],
                flush=True,
            )

    finally:
        git("checkout", "--detach", original_commit)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nKnowSWE — E6 Filtered Retrieval")
    print("=" * 75)

    for result in results:
        print(
            f"{result['id']:<35}"
            f"E6a={str(result['E6a']['hit']):<10}"
            f"E6b={str(result['E6b']['hit']):<10}"
        )

    n = len(results)

    for method in ("E6a", "E6b"):
        hits = sum(row[method]["hit"] for row in results)
        print(f"{method}: {hits}/{n} = {hits/n:.2%}")

    print("\nResults saved:", OUTPUT)


if __name__ == "__main__":
    main()
