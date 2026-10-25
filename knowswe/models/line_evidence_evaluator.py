"""Gemma evidence assessment using verified source-line references."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LineEvidenceAssessment:
    location: str
    support: float
    evidence_lines: tuple[int, ...]
    evidence_text: str
    reason: str


class GemmaLineEvidenceEvaluator:
    def __init__(self, client):
        self.client = client

    def assess(
        self,
        bug_report: str,
        location: str,
        content: str,
        start_line: int,
    ) -> LineEvidenceAssessment:

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
            "You are a software debugging assistant.\n"
            "Assess ONLY the supplied code for the reported bug.\n"
            "Return one JSON object with EXACTLY these fields:\n"
            "location, support, evidence_lines, reason.\n"
            "support must be a number between -1 and 1.\n"
            "evidence_lines must be a list of source line numbers.\n"
            "Use support 0 and evidence_lines [] if inconclusive.\n"
            "Do not invent line numbers.\n"
            "Keep reason under 25 words.\n"
            "Do not use Markdown.\n\n"
            f"BUG:\n{bug_report}\n\n"
            f"LOCATION:\n{location}\n\n"
            f"CODE:\n{numbered_code}\n"
        )

        result = self.client.generate_json(prompt)

        if not isinstance(result, dict):
            raise ValueError("Assessment must be a JSON object")

        required = {
            "location",
            "support",
            "evidence_lines",
            "reason",
        }

        if set(result) != required:
            raise ValueError("Unexpected assessment fields")

        if result["location"] != location:
            raise ValueError("Assessment location mismatch")

        support = result["support"]

        if isinstance(support, bool) or not isinstance(
            support, (int, float)
        ):
            raise ValueError("Support must be numeric")

        if not -1 <= support <= 1:
            raise ValueError("Support outside permitted range")

        evidence_lines = result["evidence_lines"]

        if not isinstance(evidence_lines, list):
            raise ValueError("Evidence lines must be a list")

        if any(
            isinstance(line, bool) or not isinstance(line, int)
            for line in evidence_lines
        ):
            raise ValueError("Evidence line numbers must be integers")

        valid_lines = set(
            range(start_line, start_line + len(source_lines))
        )

        if any(line not in valid_lines for line in evidence_lines):
            raise ValueError("Evidence references unavailable lines")

        if len(evidence_lines) != len(set(evidence_lines)):
            raise ValueError("Duplicate evidence lines")

        if support != 0 and not evidence_lines:
            raise ValueError(
                "Nonzero support requires source-line evidence"
            )

        reason = result["reason"]

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Assessment reason cannot be empty")

        evidence_text = "\n".join(
            f"{line}: {source_lines[line - start_line]}"
            for line in evidence_lines
        )

        return LineEvidenceAssessment(
            location=location,
            support=float(support),
            evidence_lines=tuple(evidence_lines),
            evidence_text=evidence_text,
            reason=reason,
        )
