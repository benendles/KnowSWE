from knowswe.retrieval.bm25 import RepositoryRetriever
from knowswe.repository.dependency_graph import RepositoryGraph


class GraphRetriever:
    def __init__(self, repository_path: str):
        self.retriever = RepositoryRetriever(repository_path)
        self.repository_graph = RepositoryGraph(repository_path)

    def index(self):
        count = self.retriever.index()
        self.repository_graph.build()
        return count

    def search(self, bug_report: str, top_k: int = 5, depth: int = 2):
        seeds = self.retriever.search(bug_report, top_k=top_k)

        candidates = {}

        for seed in seeds:
            filename = seed["file"]

            candidates[filename] = {
                "file": filename,
                "bm25_score": seed["score"],
                "source": "bm25",
                "seed_file": filename,
            }

            dependencies = self.repository_graph.dependencies(
                filename,
                depth=depth
            )

            for dependency in dependencies:
                if dependency not in candidates:
                    candidates[dependency] = {
                        "file": dependency,
                        "bm25_score": 0.0,
                        "source": "dependency",
                        "seed_file": filename,
                    }

        return list(candidates.values())
