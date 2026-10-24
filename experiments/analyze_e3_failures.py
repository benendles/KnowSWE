import json
from pathlib import Path

results_path = Path(
    "results/E4_Evaluation/multi_bug/sqlfluff_five_E3.json"
)

results = json.loads(
    results_path.read_text(encoding="utf-8")
)

print("KnowSWE — E3 Multi-Bug Failure Analysis")
print("=" * 85)

for result in results:
    e3 = result["E3"]

    print(f"\nCase: {result['id']}")
    print("Status:", "HIT" if e3["hit"] else "MISS")
    print("Reference files:", e3["reference_files"])
    print("Selected chunks:", len(e3["selected_chunks"]))
    print("Tokens used:", e3["estimated_tokens_used"])

    selected_files = {
        chunk["file"]
        for chunk in e3["selected_chunks"]
    }

    print("Unique selected files:", len(selected_files))

    print("\nTop five selected chunks:")

    for chunk in e3["selected_chunks"][:5]:
        print(
            f"  {chunk['file']}:"
            f"{chunk['start_line']}-{chunk['end_line']}"
            f" | tokens={chunk['estimated_tokens']}"
            f" | score={chunk['score']:.3f}"
        )

    print("-" * 85)
