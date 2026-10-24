import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASET = "princeton-nlp/SWE-bench_Lite"

output_dir = Path("data/benchmarks/swebench_lite")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading SWE-bench Lite development dataset...")

parquet_path = hf_hub_download(
    repo_id=DATASET,
    filename="data/dev-00000-of-00001.parquet",
    repo_type="dataset",
)

table = pq.read_table(parquet_path)
rows = table.to_pylist()

cases = []

for row in rows:
    patch = row["patch"]

    # Extract files modified by the reference fix.
    changed_files = re.findall(
        r"^\+\+\+ b/(.+)$",
        patch,
        flags=re.MULTILINE,
    )

    changed_files = sorted(set(changed_files))

    cases.append({
        "id": row["instance_id"],
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "bug_report": row["problem_statement"],
        "fix_files": changed_files,
        "patch": patch,
    })

destination = output_dir / "dev_labeled.json"

destination.write_text(
    json.dumps(cases, indent=2),
    encoding="utf-8",
)

print("\nKnowSWE — Historical Bug Dataset")
print("=" * 60)
print("Total development cases:", len(cases))
print("Unique repositories:", len(set(c["repo"] for c in cases)))
print("Cases with fix-file labels:", sum(bool(c["fix_files"]) for c in cases))

print("\nFirst five cases:")

for case in cases[:5]:
    print(f"\nID: {case['id']}")
    print(f"Repository: {case['repo']}")
    print(f"Base commit: {case['base_commit']}")
    print(f"Fix files: {case['fix_files']}")

print("\nDataset saved to:", destination)
