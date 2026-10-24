import re
from pathlib import Path

from rank_bm25 import BM25Okapi


SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx"
}

IGNORED_DIRECTORIES = {
    ".git", ".venv", "node_modules",
    "__pycache__", "dist", "build",
    ".next", ".pytest_cache"
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class RepositoryRetriever:
    def __init__(self, repository_path: str):
        self.repository_path = Path(repository_path).resolve()
        self.files = []
        self.documents = []
        self.bm25 = None

    def index(self):
        if not self.repository_path.is_dir():
            raise ValueError("Repository path does not exist")

        self.files.clear()
        self.documents.clear()

        for path in sorted(self.repository_path.rglob("*")):
            if not path.is_file():
                continue

            relative_path = path.relative_to(self.repository_path)

            if any(
                part in IGNORED_DIRECTORIES
                for part in relative_path.parts
            ):
                continue

            if path.suffix not in SUPPORTED_EXTENSIONS:
                continue

            if path.stat().st_size > 500_000:
                continue

            try:
                content = path.read_text(
                    encoding="utf-8",
                    errors="replace"
                )
            except OSError:
                continue

            if not content.strip():
                continue

            self.files.append(str(relative_path))

            self.documents.append(
                tokenize(f"{relative_path} {content}")
            )

        if not self.documents:
            raise ValueError("No supported source files found")

        self.bm25 = BM25Okapi(self.documents)

        return len(self.files)

    def search(self, bug_report: str, top_k: int = 5):
        if self.bm25 is None:
            raise RuntimeError("Call index() before search()")

        scores = self.bm25.get_scores(tokenize(bug_report))

        ranked = sorted(
            zip(self.files, scores),
            key=lambda item: (-item[1], item[0])
        )

        return [
            {
                "file": filename,
                "score": float(score)
            }
            for filename, score in ranked[:top_k]
        ]
