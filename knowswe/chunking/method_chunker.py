"""Method-aware Python source chunking for KnowSWE BEDS-3."""

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MethodChunk:
    filename: str
    start_line: int
    end_line: int
    symbol: str
    parent_symbol: str | None
    content: str
    estimated_tokens: int

    @property
    def chunk_id(self) -> str:
        return (
            f"{self.filename}:"
            f"{self.start_line}:{self.end_line}"
        )


def estimate_tokens(text: str) -> int:
    """Approximate token count; not model-tokenizer exact."""
    return max(1, (len(text) + 3) // 4)


def chunk_python_methods(
    path: Path,
    repository_root: Path,
    max_tokens: int = 300,
) -> list[MethodChunk]:
    """Split Python source into method-aware, line-bounded chunks."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    root = Path(repository_root).resolve()
    path = Path(path).resolve()

    if not path.is_relative_to(root):
        raise ValueError("Source file is outside repository root")

    source = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    filename = path.relative_to(root).as_posix()

    regions: list[tuple[int, int, str, str | None]] = []

    def add_region(
        start: int,
        end: int,
        symbol: str,
        parent: str | None,
    ):
        if start <= end:
            regions.append((start, end, symbol, parent))

    def process_class(
        node: ast.ClassDef,
        class_start: int,
    ):
        children = sorted(
            (
                child
                for child in node.body
                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                    ),
                )
            ),
            key=lambda child: child.lineno,
        )

        cursor = class_start

        for child in children:
            start = min(
                getattr(child, "decorator_list", [])
                or [child],
                key=lambda item: item.lineno,
            ).lineno

            if cursor < start:
                add_region(
                    cursor,
                    start - 1,
                    node.name,
                    None,
                )

            if isinstance(child, ast.ClassDef):
                add_region(
                    start,
                    child.end_lineno,
                    f"{node.name}.{child.name}",
                    node.name,
                )
            else:
                add_region(
                    start,
                    child.end_lineno,
                    f"{node.name}.{child.name}",
                    node.name,
                )

            cursor = child.end_lineno + 1

        if cursor <= node.end_lineno:
            add_region(
                cursor,
                node.end_lineno,
                node.name,
                None,
            )

    top_level = sorted(
        (
            node
            for node in tree.body
            if isinstance(
                node,
                (
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
        ),
        key=lambda node: node.lineno,
    )

    cursor = 1

    for node in top_level:
        start = min(
            getattr(node, "decorator_list", [])
            or [node],
            key=lambda item: item.lineno,
        ).lineno

        if cursor < start:
            add_region(
                cursor,
                start - 1,
                "<module>",
                None,
            )

        if isinstance(node, ast.ClassDef):
            process_class(node, start)
        else:
            add_region(
                start,
                node.end_lineno,
                node.name,
                None,
            )

        cursor = node.end_lineno + 1

    if cursor <= len(lines):
        add_region(
            cursor,
            len(lines),
            "<module>",
            None,
        )

    chunks: list[MethodChunk] = []

    for start, end, symbol, parent in regions:
        current_start = start
        buffer = ""

        for line_number in range(start, end + 1):
            line = lines[line_number - 1]

            proposed = buffer + line

            if buffer and estimate_tokens(proposed) > max_tokens:
                chunks.append(
                    MethodChunk(
                        filename=filename,
                        start_line=current_start,
                        end_line=line_number - 1,
                        symbol=symbol,
                        parent_symbol=parent,
                        content=buffer,
                        estimated_tokens=estimate_tokens(buffer),
                    )
                )

                current_start = line_number
                buffer = line
            else:
                buffer = proposed

        if buffer:
            chunks.append(
                MethodChunk(
                    filename=filename,
                    start_line=current_start,
                    end_line=end,
                    symbol=symbol,
                    parent_symbol=parent,
                    content=buffer,
                    estimated_tokens=estimate_tokens(buffer),
                )
            )

    return chunks
