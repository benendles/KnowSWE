"""Minimal source-grounded evidence assessment using Gemma."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MinimalEvidenceAssessment:
    location: str
    support: int
    evidence_line: int | None
    evidence_text: str | None


class MinimalGemmaEvidenceEvaluator:
    """Ask Gemma for a signed assessment and one source-line reference."""

    def __init__(self, client):
        self.client = client

    def assess(
        self,
        bug_report: str,
        location: str,
        content: str,
        start_line: int,
    ) -> MinimalEvidenceAssessment:

        if not content.strip():
            raise ValueError("Cannot assess empty source code")

        if start_line < 1:
            raise ValueError("Invalid starting line")

        source_lines = content.splitlines()

        numbered_code = "\n".join(
            f"{start_line + index}: {line}"
            for index, line in enumerate(source_lines)
        )

        prompt = (
            "You are checking whether a code fragment may contain a bug.\n"
            "Return ONLY a JSON object with two fields:\n"
            '"support": one of -1, 0, 1\n'
            '"line": an integer source line number, or null\n\n'
            "Use support 1 if the code contains suspicious logic.\n"
            "Use support -1 only if the code provides evidence against "
            "the suspected fault being here.\n"
            "Use support 0 if the evidence is inconclusive.\n"
            "If support is 0, use line null.\n"
            "Do not copy source code into the JSON.\n"
            "Do not include explanations or Markdown.\n\n"
            "Example output:\n"
            '{"support": 0, "line": null}\n\n'
            f"BUG: {bug_report}\n\n"
            f"CODE:\n{numbered_code}\n"
        )

        result = self.client.generate_json(prompt)

        if not isinstance(result, dict):
            raise ValueError("Assessment must be a JSON object")

        if set(result) != {"support", "line"}:
            raise ValueError("Unexpected assessment fields")

        support = result["support"]
        line = result["line"]

        if type(support) is not int or support not in (-1, 0, 1):
            raise ValueError("Invalid support value")

        valid_lines = range(
            start_line,
            start_line + len(source_lines),
        )

        if line is not None and (
            type(line) is not int or line not in valid_lines
        ):
            raise ValueError("Invalid evidence line")

        if support == 0 and line is not None:
            raise ValueError(
                "Neutral assessment must have no evidence line"
            )

        if support != 0 and line is None:
            raise ValueError(
                "Nonzero support requires an evidence line"
            )

        evidence_text = (
            source_lines[line - start_line]
            if line is not None
            else None
        )

        return MinimalEvidenceAssessment(
            location=location,
            support=support,
            evidence_line=line,
            evidence_text=evidence_text,
        )
