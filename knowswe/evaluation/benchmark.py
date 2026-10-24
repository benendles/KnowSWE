import json
from pathlib import Path

from knowswe.beds.controller import Candidate, BEDSController
from knowswe.beds.inspection import estimate_tokens
from knowswe.repository.dependency_graph import RepositoryGraph
from knowswe.retrieval.bm25 import RepositoryRetriever


def file_cost(root, filename):
    path = (root / filename).resolve()

    if not path.is_relative_to(root) or not path.is_file():
        return None

    content = path.read_text(encoding="utf-8", errors="replace")
    return estimate_tokens(content)


def select_with_budget(ordered_files, costs, budget):
    selected = []
    remaining = budget

    for filename in ordered_files:
        if filename in selected:
            continue

        cost = costs.get(filename)

        if cost is None or cost <= 0:
            continue

        if cost <= remaining:
            selected.append(filename)
            remaining -= cost

    return selected, remaining


def evaluate_case(case, budget=100, top_k=1, depth=2):
    root = Path(case["repository"]).resolve()

    retriever = RepositoryRetriever(str(root))
    retriever.index()

    graph = RepositoryGraph(str(root))
    graph.build()

    bug_report = case["bug_report"]
    faulty_files = set(case["faulty_files"])

    # Rank all indexed files once.
    ranked = retriever.search(
        bug_report,
        top_k=len(retriever.files),
    )

    ranked_names = [item["file"] for item in ranked]

    # Use one shared seed set for E1 and E2.
    seeds = ranked[:top_k]

    distances = {}
    relevance = {}

    for seed in seeds:
        filename = seed["file"]
        relevance[filename] = seed["score"]

        distances[filename] = min(
            distances.get(filename, float("inf")),
            0,
        )

        dependencies = graph.dependency_distances(
            filename,
            depth=depth,
        )

        for dependency, distance in dependencies.items():
            distances[dependency] = min(
                distances.get(dependency, float("inf")),
                distance,
            )

    # Shared cost accounting for all configurations.
    costs = {
        filename: file_cost(root, filename)
        for filename in set(ranked_names) | set(distances)
    }

    # E0: BM25 ranking only.
    e0_files, e0_remaining = select_with_budget(
        ranked_names,
        costs,
        budget,
    )

    # E1: BM25 seeds, then graph dependencies.
    # Order by dependency distance and filename.
    e1_order = [item["file"] for item in seeds]

    graph_candidates = sorted(
        (
            (filename, distance)
            for filename, distance in distances.items()
            if filename not in e1_order
        ),
        key=lambda item: (item[1], item[0]),
    )

    e1_order.extend(filename for filename, _ in graph_candidates)

    e1_files, e1_remaining = select_with_budget(
        e1_order,
        costs,
        budget,
    )

    # E2: Same graph candidate pool, BEDS ordering.
    controller = BEDSController(budget=budget)

    candidates = [
        Candidate(
            filename=filename,
            relevance=relevance.get(filename, 0.0),
            distance=distance,
            cost=costs[filename],
        )
        for filename, distance in distances.items()
        if costs.get(filename) is not None
    ]

    e2_files = []

    while candidates:
        selected = controller.select(candidates)

        if selected is None:
            break

        e2_files.append(selected.filename)
        candidates.remove(selected)

    e2_remaining = controller.remaining

    results = {}

    for name, selected, remaining in [
        ("E0_BM25", e0_files, e0_remaining),
        ("E1_Graph", e1_files, e1_remaining),
        ("E2_BEDS", e2_files, e2_remaining),
    ]:
        found = faulty_files.intersection(selected)

        results[name] = {
            "selected_files": selected,
            "faulty_files_found": sorted(found),
            "hit": bool(found),
            "recall": len(found) / len(faulty_files) if faulty_files else 0.0,
            "estimated_tokens_used": budget - remaining,
            "remaining_budget": remaining,
        }

    return {
        "id": case["id"],
        "bug_report": bug_report,
        "faulty_files": sorted(faulty_files),
        "budget": budget,
        "results": results,
    }


def evaluate_dataset(dataset_path, budget=100, top_k=1, depth=2):
    cases = json.loads(
        Path(dataset_path).read_text(encoding="utf-8")
    )

    case_results = [
        evaluate_case(
            case,
            budget=budget,
            top_k=top_k,
            depth=depth,
        )
        for case in cases
    ]

    summary = {}

    for method in ["E0_BM25", "E1_Graph", "E2_BEDS"]:
        rows = [
            case["results"][method]
            for case in case_results
        ]

        count = len(rows)

        summary[method] = {
            "cases": count,
            "hit_rate": (
                sum(row["hit"] for row in rows) / count
                if count else 0.0
            ),
            "mean_recall": (
                sum(row["recall"] for row in rows) / count
                if count else 0.0
            ),
            "mean_estimated_tokens": (
                sum(row["estimated_tokens_used"] for row in rows) / count
                if count else 0.0
            ),
        }

    return {
        "dataset": dataset_path,
        "budget": budget,
        "top_k": top_k,
        "depth": depth,
        "summary": summary,
        "cases": case_results,
    }
