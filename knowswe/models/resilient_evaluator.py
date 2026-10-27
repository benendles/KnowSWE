"""Fault-tolerant wrapper for evidence assessment."""

from dataclasses import dataclass, field
from time import perf_counter


@dataclass(frozen=True)
class InferenceAttempt:
    attempt: int
    success: bool
    elapsed_seconds: float
    error: str | None = None


@dataclass
class ResilientAssessment:
    assessment: object | None
    attempts: list[InferenceAttempt] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.assessment is not None

    @property
    def total_seconds(self) -> float:
        return sum(a.elapsed_seconds for a in self.attempts)


class ResilientEvidenceEvaluator:
    """Retry invalid assessments while preserving an audit trail."""

    def __init__(self, evaluator, max_retries: int = 1):
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")

        self.evaluator = evaluator
        self.max_retries = max_retries

    def assess(self, **kwargs) -> ResilientAssessment:
        attempts = []

        for attempt_number in range(1, self.max_retries + 2):
            started = perf_counter()

            try:
                assessment = self.evaluator.assess(**kwargs)

            except (ValueError, RuntimeError, TimeoutError) as exc:
                attempts.append(
                    InferenceAttempt(
                        attempt=attempt_number,
                        success=False,
                        elapsed_seconds=perf_counter() - started,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            attempts.append(
                InferenceAttempt(
                    attempt=attempt_number,
                    success=True,
                    elapsed_seconds=perf_counter() - started,
                )
            )

            return ResilientAssessment(
                assessment=assessment,
                attempts=attempts,
            )

        return ResilientAssessment(
            assessment=None,
            attempts=attempts,
        )
