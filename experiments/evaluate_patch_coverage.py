import json
import re
from pathlib import Path


DATASET = Path(
    "data/benchmarks/swebench_lite/dev_labeled.json"
)

E3_PATH = Path(
    "results/E4_Evaluation/multi_bug/sqlfluff_five_E3.json"
)

E5_PATH = Path(
    "results/E4_Evaluation/multi_bug/sqlfluff_five_E3_vs_E5.json"
)

OUTPUT = Path(
    "results/E4_Evaluation/multi_bug/"
    "sqlfluff_five_patch_coverage.json"
)


def extract_changed_lines(patch):
    changed = {}

    current_file = None
    old_line = None

    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue

        if line.startswith("@@"):
            match = re.search(r"@@ -(\d+)", line)
            old_line = int(match.group(1)) if match else None
            continue

        if current_file is None or old_line is None:
            continue

        if line.startswith("-") and not line.startswith("---"):
            changed.setdefault(current_file, set()).add(old_line)
            old_line += 1

        elif line.startswith("+") and not line.startswith("+++"):
            continue

        elif line.startswith(" "):
            old_line += 1

    return changed


def coverage(chunks, changed):
    total = sum(len(lines) for lines in changed.values())

    covered = 0

    for filename, lines in changed.items():
        relevant = [
            chunk for chunk in chunks
            if chunk["file"] == filename
        ]

        for line in lines:
            if any(
                chunk["start_line"] <= line <= chunk["end_line"]
                for chunk in relevant
            ):
                covered += 1

    return {
        "covered_lines": covered,
        "total_changed_lines": total,
        "coverage": covered / total if total else None,
    }


def main():
    dataset = {
        row["id"]: row
        for row in json.loads(DATASET.read_text())
    }

    e3_results = json.loads(E3_PATH.read_text())
    e5_results = json.loads(E5_PATH.read_text())

    e3_lookup = {
        row["id"]: row["E3"]
        for row in e3_results
    }

    results = []

    print("KnowSWE — Patch-Line Coverage")
    print("=" * 80)

    for row in e5_results:
        case_id = row["id"]

        changed = extract_changed_lines(
            dataset[case_id]["patch"]
        )

        e3 = coverage(
            e3_lookup[case_id]["selected_chunks"],
            changed,
        )

        e5 = coverage(
            row["E5"]["selected_chunks"],
            changed,
        )

        results.append({
            "id": case_id,
            "E3": e3,
            "E5": e5,
        })

        def format_coverage(result):
            if result["coverage"] is None:
                return "N/A"
            return f"{result['coverage']:.1%}"

        print(
            f"{case_id:<35}"
            f"E3={format_coverage(e3):<12}"
            f"E5={format_coverage(e5):<12}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nResults saved:", OUTPUT)


if __name__ == "__main__":
    main()
