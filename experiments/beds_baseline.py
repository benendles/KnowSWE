import argparse
import json

from knowswe.beds.inspection import BudgetedRepositoryInspector


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo", required=True)
    parser.add_argument("--bug", required=True)
    parser.add_argument("--budget", type=int, default=1000)

    args = parser.parse_args()

    inspector = BudgetedRepositoryInspector(
        repository_path=args.repo,
        budget=args.budget,
    )

    result = inspector.investigate(args.bug)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
