import argparse
import json

from knowswe.retrieval.bm25 import RepositoryRetriever


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo", required=True)
    parser.add_argument("--bug", required=True)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    retriever = RepositoryRetriever(args.repo)

    indexed = retriever.index()
    results = retriever.search(args.bug, args.top_k)

    print(json.dumps({
        "indexed_files": indexed,
        "bug_report": args.bug,
        "ranked_files": results
    }, indent=2))


if __name__ == "__main__":
    main()
