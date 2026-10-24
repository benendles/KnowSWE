import json
import subprocess
from pathlib import Path

from experiments.evaluate_chunk_beds import evaluate as evaluate_e5


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
    "sqlfluff_five_E3_vs_E5.json"
)


def git(*args):
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main():
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))
    previous = json.loads(E3_RESULTS.read_text(encoding="utf-8"))

    e3_lookup = {
        item["id"]: item["E3"]
        for item in previous
    }

    if git("status", "--porcelain"):
        raise RuntimeError("SQLFluff repository contains local changes.")

    original_commit = git("rev-parse", "HEAD")
    results = []

    try:
        for index, case in enumerate(cases, 1):
            print(
                f"\n[{index}/{len(cases)}] {case['id']}",
                flush=True,
            )

            git("checkout", "--detach", case["base_commit"])

            benchmark_case = {
                "id": case["id"],
                "repository": str(ROOT),
                "bug_report": case["bug_report"],
                "faulty_files": case["faulty_files"],
            }

            e5 = evaluate_e5(
                benchmark_case,
                budget=2000,
                chunk_size=500,
            )

            e3 = e3_lookup[case["id"]]

            results.append({
                "id": case["id"],
                "base_commit": case["base_commit"],
                "E3": {
                    "hit": e3["hit"],
                    "tokens": e3["estimated_tokens_used"],
                    "chunks": len(e3["selected_chunks"]),
                },
                "E5": {
                    "hit": e5["hit"],
                    "tokens": e5["estimated_tokens_used"],
                    "chunks": len(e5["selected_chunks"]),
                    "selected_chunks": e5["selected_chunks"],
                },
            })

            print(
                f"E3={e3['hit']} | E5={e5['hit']} | "
                f"E5 chunks={len(e5['selected_chunks'])}",
                flush=True,
            )

    finally:
        git("checkout", "--detach", original_commit)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nKnowSWE — E3 vs E5")
    print("=" * 85)

    print(
        f"{'Case':<35}"
        f"{'E3':<12}"
        f"{'E5':<12}"
        f"{'E5 Chunks':<12}"
    )

    print("-" * 85)

    for item in results:
        print(
            f"{item['id']:<35}"
            f"{str(item['E3']['hit']):<12}"
            f"{str(item['E5']['hit']):<12}"
            f"{item['E5']['chunks']:<12}"
        )

    n = len(results)
    e3_hits = sum(item["E3"]["hit"] for item in results)
    e5_hits = sum(item["E5"]["hit"] for item in results)

    print("-" * 85)
    print(f"E3 hit rate: {e3_hits}/{n} = {e3_hits/n:.2%}")
    print(f"E5 hit rate: {e5_hits}/{n} = {e5_hits/n:.2%}")
    print(f"Difference: {(e5_hits-e3_hits)/n:+.2%}")
    print("\nResults saved:", OUTPUT)


if __name__ == "__main__":
    main()
