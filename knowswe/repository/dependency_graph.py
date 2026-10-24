import ast
from pathlib import Path

import networkx as nx


class RepositoryGraph:
    def __init__(self, repository_path):
        self.root = Path(repository_path).resolve()
        self.graph = nx.DiGraph()

    def build(self):
        self.graph.clear()

        files = {}

        for path in self.root.rglob("*.py"):
            if any(
                part.startswith(".")
                or part in {"node_modules", "__pycache__"}
                for part in path.relative_to(self.root).parts
            ):
                continue

            relative = path.relative_to(self.root).as_posix()
            files[relative] = path
            self.graph.add_node(relative)

        modules = {}

        for filename in files:
            parts = Path(filename).with_suffix("").parts

            if parts[-1] == "__init__":
                module = ".".join(parts[:-1])
            else:
                module = ".".join(parts)

            if module:
                modules[module] = filename

        def resolve_module(module):
            if module in modules:
                return modules[module]

            return None

        for filename, path in files.items():
            try:
                tree = ast.parse(
                    path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                )
            except (SyntaxError, UnicodeError):
                continue

            parts = Path(filename).with_suffix("").parts

            if parts[-1] == "__init__":
                package_parts = list(parts[:-1])
            else:
                package_parts = list(parts[:-1])

            for node in ast.walk(tree):
                targets = []

                if isinstance(node, ast.Import):
                    targets = [
                        alias.name
                        for alias in node.names
                    ]

                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        keep = len(package_parts) - node.level + 1

                        if keep < 0:
                            continue

                        base_parts = package_parts[:keep]

                        if node.module:
                            base_parts.extend(node.module.split("."))

                        base = ".".join(base_parts)

                    else:
                        base = node.module or ""

                    if base:
                        targets.append(base)

                    for alias in node.names:
                        if alias.name == "*":
                            continue

                        candidate = (
                            f"{base}.{alias.name}"
                            if base
                            else alias.name
                        )

                        targets.append(candidate)

                for module in targets:
                    target = resolve_module(module)

                    if target and target != filename:
                        self.graph.add_edge(filename, target)

        return self.graph

    def dependencies(self, filename, depth=2):
        return sorted(
            self.dependency_distances(filename, depth).keys()
        )

    def dependency_distances(self, filename, depth=2):
        if filename not in self.graph:
            return {}

        lengths = nx.single_source_shortest_path_length(
            self.graph,
            filename,
            cutoff=depth,
        )

        return {
            name: distance
            for name, distance in sorted(
                lengths.items(),
                key=lambda item: (item[1], item[0]),
            )
            if name != filename
        }
