import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeChunk:
    filename: str
    start_line: int
    end_line: int
    symbol: str
    content: str
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def chunk_python_file(
    path: Path,
    repository_root: Path,
    max_tokens: int = 500,
) -> list[CodeChunk]:

    source = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = source.splitlines(keepends=True)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    filename = path.relative_to(repository_root).as_posix()

    chunks = []

    def add_region(start: int, end: int, symbol: str):
        if start > end:
            return

        current_start = start

        while current_start <= end:
            current_end = current_start
            content = ""

            while current_end <= end:
                candidate = content + lines[current_end - 1]

                if (
                    estimate_tokens(candidate) > max_tokens
                    and content
                ):
                    break

                content = candidate
                current_end += 1

                if estimate_tokens(content) >= max_tokens:
                    break

            if not content:
                break

            chunks.append(
                CodeChunk(
                    filename=filename,
                    start_line=current_start,
                    end_line=current_end - 1,
                    symbol=symbol,
                    content=content,
                    estimated_tokens=estimate_tokens(content),
                )
            )

            current_start = current_end

    if tree is None:
        add_region(1, len(lines), "<syntax_fallback>")
        return chunks

    regions = []

    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            regions.append(
                (node.lineno, node.end_lineno, node.name)
            )

        elif isinstance(node, ast.ClassDef):
            regions.append(
                (node.lineno, node.end_lineno, node.name)
            )

    if not regions:
        add_region(1, len(lines), "<module>")
        return chunks

    cursor = 1

    for start, end, symbol in sorted(regions):
        if cursor < start:
            add_region(cursor, start - 1, "<module>")

        add_region(start, end, symbol)
        cursor = end + 1

    if cursor <= len(lines):
        add_region(cursor, len(lines), "<module>")

    return chunks
