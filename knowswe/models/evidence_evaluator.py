"""Validate Gemma-generated assessments of inspected source code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceAssessment:
    location: str
    support: float
    evidence_quote: str
    reason: str


class GemmaEvidenceEvaluator:
    def __init__(self, client):
        self.client = client

    def assess(self, bug_report: str, location: str, content: str):
        if not content.strip():
            raise ValueError("Cannot assess empty source code")

        prompt = (
            "You are investigating a software bug.\n"
            "Evaluate ONLY the supplied source code.\n"
            "Do not claim to have inspected other files.\n"
            "A support score measures how strongly this code "
            "suggests the suspected fault is here.\n"
            "Use -1 for contradicting evidence, "
            "0 for inconclusive evidence, "
            "and 1 for strong supporting evidence.\n"
            "Return exactly one JSON object with fields: "
            "location, support, evidence_quote, reason.\n"
            "The evidence_quote must be an exact substring "
            "of the supplied source code.\n"
            "If no specific evidence exists, use an empty "
            "evidence_quote and support 0.\n\n"
            f"BUG REPORT:\n{bug_report}\n\n"
            f"LOCATION:\n{location}\n\n"
            f"SOURCE CODE:\n{content}\n"
        )

        result = self.client.generate_json(prompt)

        if not isinstance(result, dict):
            raise ValueError("Assessment must be a JSON object")

        required = {
            "location",
            "support",
            "evidence_quote",
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

        if not -1.0 <= support <= 1.0:
            raise ValueError("Support outside permitted range")

        quote = result["evidence_quote"]
        reason = result["reason"]

        if not isinstance(quote, str):
            raise ValueError("Evidence quote must be text")

        if not isinstance(reason, str):
            raise ValueError("Reason must be text")

        if quote and quote not in content:
            raise ValueError("Evidence quote not found in source")

        if not quote and support != 0:
            raise ValueError(
                "Nonzero support requires source evidence"
            )

        if not reason.strip():
            raise ValueError("Assessment reason cannot be empty")

        return EvidenceAssessment(
            location=location,
            support=float(support),
            evidence_quote=quote,
            reason=reason,
        )
