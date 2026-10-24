import json
import re
import subprocess
from pathlib import Path

from rank_bm25 import BM25Okapi
from knowswe.chunking.ast_chunker import chunk_python_file


def tokenize(text):
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", text.lower())


repo = Path("data/real_repos/sqlfluff").resolve()

manifest = json.loads(
    Path(
        "data/benchmarks/swebench_lite/multi_bug/"
        "sqlfluff_five_cases.json"
    ).read_text()
)

results = json.loads(
    Path(
        "results/E4_Evaluation/multi_bug/"
        "sqlfluff_five_E3.json"
    ).read_text()
)

missed = {
    result["id"]
    for result in results
    if not result["E3"]["hit"]
}


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


if git("status", "--porcelain"):
    raise RuntimeError("Repository has uncommitted changes.")

original_commit = git("rev-parse", "HEAD")

try:
    for case in manifest:
        if case["id"] not in missed:
            continue

        git("checkout", "--detach", case["base_commit"])

        chunks = []

        for path in sorted(repo.rglob("*.py")):
            if ".git" in path.parts or ".venv" in path.parts:
                continue

            chunks.extend(
                chunk_python_file(
                    path,
                    repo,
                    max_tokens=500,
                )
            )

        corpus = [
            tokenize(
                chunk.filename + " " +
                chunk.symbol + " " +
                chunk.content
            )
            for chunk in chunks
        ]

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenize(case["bug_report"]))

        ranking = sorted(
            range(len(chunks)),
            key=lambda i: (-scores[i], i),
        )

        references = set(case["faulty_files"])

        matching = [
            (rank, chunks[index], scores[index])
            for rank, index in enumerate(ranking, 1)
            if chunks[index].filename in references
        ]

        print("\n" + "=" * 75)
        print("Case:", case["id"])
        print("Reference files:", sorted(references))
        print("Indexed chunks:", len(chunks))

        if matching:
            print("Best reference chunk rank:", matching[0][0])
            print("Best reference chunk score:", round(float(matching[0][2]), 3))
            print("Best reference chunk lines:",
                  matching[0][1].start_line,
                  "-",
                  matching[0][1].end_line)
        else:
            print("No reference-file chunks indexed.")

finally:
    git("checkout", "--detach", original_commit)
