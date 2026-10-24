import json
import subprocess
from pathlib import Path

from experiments.evaluate_hierarchical import evaluate as evaluate_e4


ROOT = Path("data/real_repos/sqlfluff").resolve()

MANIFEST = Path(
    "data/benchmarks/swebench_lite/multi_bug/"
    "sqlfluff_five_cases.json"
)

E3_RESULTS = Path(
    "results/E4_Evaluation/multi_bug/"
    "sqlfluff_five_E3.json"
)

OUTPUT = Path(
    "results/E4_Evaluation/multi_bug/"
    "sqlfluff_five_E3_vs_E4.json"
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
    cases = json.loads(MANIFEST.read_text())
    previous = json.loads(E3_RESULTS.read_text())

    e3_lookup = {
        item["id"]: item["E3"]
        for item in previous
    }

    original_commit = git("rev-parse", "HEAD")

    if git("status", "--porcelain"):
        raise RuntimeError(
            "Repository contains uncommitted changes."
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

            e4 = evaluate_e4(
                benchmark_case,
                budget=2000,
                chunk_size=500,
                file_weight=0.4,
            )

            e3 = e3_lookup[case["id"]]

            results.append({
                "id": case["id"],
                "base_commit": case["base_commit"],
                "E3": {
                    "hit": e3["hit"],
                    "tokens": e3["estimated_tokens_used"],
                },
                "E4": {
                    "hit": e4["hit"],
                    "tokens": e4["estimated_tokens_used"],
                    "selected_chunks": e4["selected_chunks"],
                },
            })

            print(
                f"E3={e3['hit']} | "
                f"E4={e4['hit']}",
                flush=True,
            )

    finally:
        git("checkout", "--detach", original_commit)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nKnowSWE — E3 vs E4 Comparison")
    print("=" * 75)

    print(
        f"{'Case':<35}"
        f"{'E3':<12}"
        f"{'E4':<12}"
    )

    print("-" * 75)

    for item in results:
        print(
            f"{item['id']:<35}"
            f"{str(item['E3']['hit']):<12}"
            f"{str(item['E4']['hit']):<12}"
        )

    e3_hits = sum(item["E3"]["hit"] for item in results)
    e4_hits = sum(item["E4"]["hit"] for item in results)

    print("-" * 75)

    print(
        f"E3 hit rate: {e3_hits}/{len(results)} "
        f"= {e3_hits / len(results):.2%}"
    )

    print(
        f"E4 hit rate: {e4_hits}/{len(results)} "
        f"= {e4_hits / len(results):.2%}"
    )

    print("\nResults saved:", OUTPUT)


if __name__ == "__main__":
    main()
