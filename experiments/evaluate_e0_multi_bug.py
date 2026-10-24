import json
import subprocess
from pathlib import Path

from knowswe.evaluation.benchmark import evaluate_case


ROOT = Path("data/real_repos/sqlfluff").resolve()

MANIFEST = Path(
    "data/benchmarks/swebench_lite/multi_bug/"
    "sqlfluff_five_cases.json"
)

OUTPUT = Path(
    "results/E4_Evaluation/multi_bug/"
    "sqlfluff_five_E0.json"
)


def git(*args):
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main():
    cases = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    original_commit = git("rev-parse", "HEAD")

    if git("status", "--porcelain"):
        raise RuntimeError(
            "SQLFluff checkout contains uncommitted changes."
        )

    results = []

    try:
        for index, case in enumerate(cases, 1):
            print(
                f"\n[{index}/{len(cases)}] {case['id']}",
                flush=True,
            )

            git(
                "checkout",
                "--detach",
                case["base_commit"],
            )

            benchmark_case = {
                "id": case["id"],
                "repository": str(ROOT),
                "bug_report": case["bug_report"],
                "faulty_files": case["faulty_files"],
            }

            evaluation = evaluate_case(
                benchmark_case,
                budget=2000,
                top_k=1,
                depth=2,
            )

            e0 = evaluation["results"]["E0_BM25"]

            results.append({
                "id": case["id"],
                "base_commit": case["base_commit"],
                "E0": e0,
            })

            print(
                f"E0 hit={e0['hit']} "
                f"tokens={e0['estimated_tokens_used']}",
                flush=True,
            )

    finally:
        git("checkout", "--detach", original_commit)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nKnowSWE — E0 Multi-Bug Baseline")
    print("=" * 75)

    print(
        f"{'Case':<35}"
        f"{'Hit':<12}"
        f"{'Tokens':<12}"
    )

    print("-" * 75)

    for item in results:
        e0 = item["E0"]

        print(
            f"{item['id']:<35}"
            f"{str(e0['hit']):<12}"
            f"{e0['estimated_tokens_used']:<12}"
        )

    hits = sum(item["E0"]["hit"] for item in results)

    print("-" * 75)

    print(
        f"E0 hit rate: {hits}/{len(results)} "
        f"= {hits / len(results):.2%}"
    )

    print("\nResults saved:", OUTPUT)


if __name__ == "__main__":
    main()
