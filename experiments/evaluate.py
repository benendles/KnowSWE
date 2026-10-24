import argparse
import json
from pathlib import Path

from knowswe.evaluation.benchmark import evaluate_dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="data/benchmarks/toy_bugs.json",
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output",
        default="results/E4_Evaluation/evaluation.json",
    )

    args = parser.parse_args()

    result = evaluate_dataset(
        dataset_path=args.dataset,
        budget=args.budget,
        top_k=args.top_k,
        depth=args.depth,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nKnowSWE Automated Evaluation")
    print("=" * 65)

    print(f"Dataset: {args.dataset}")
    print(f"Budget: {args.budget} estimated tokens")

    print()
    print(
        f"{'Method':<15}"
        f"{'Hit Rate':>12}"
        f"{'Recall':>12}"
        f"{'Avg Tokens':>15}"
    )

    print("-" * 65)

    for method, metrics in result["summary"].items():
        print(
            f"{method:<15}"
            f"{metrics['hit_rate']:>12.2%}"
            f"{metrics['mean_recall']:>12.2%}"
            f"{metrics['mean_estimated_tokens']:>15.1f}"
        )

    print(f"\nDetailed results saved to: {output}")


if __name__ == "__main__":
    main()
