"""Response grading for benchmark tasks.

Evaluates agent outputs against expected keyword sets to measure
accuracy and detect silent failures (wrong answers without errors).

Usage::

    from agentprobe.benchmark.grader import ResponseGrader

    grader = ResponseGrader(threshold=0.5)
    result = grader.grade("Tokyo has 14 million people", ["tokyo", "population", "million"])
    # result.score = 1.0, result.is_correct = True

    # Silent failure detection
    batch = grader.grade_batch(
        outputs=["Paris is the capital", "I don't know"],
        expected=[["tokyo", "population"], ["berlin", "capital"]],
        had_errors=[False, False],
    )
    # batch.silent_failure_count = 2 (wrong answers, no exceptions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class GradeResult:
    """Result of grading a single agent response."""

    score: float  # 0.0-1.0, fraction of keywords matched
    matched_keywords: tuple[str, ...]
    total_keywords: int
    is_correct: bool  # score >= threshold


@dataclass(frozen=True)
class BatchGradeResult:
    """Aggregated grading results for a batch of responses."""

    n_total: int
    n_correct: int
    n_incorrect: int
    accuracy: float  # n_correct / n_total
    mean_score: float
    silent_failure_count: int  # incorrect AND no exception raised
    silent_failure_rate: float  # silent_failure_count / n_total
    grades: tuple[GradeResult, ...]


class ResponseGrader:
    """Grade agent responses by keyword matching.

    Args:
        threshold: Minimum keyword match fraction to count as correct.
            Default 0.5 means at least half the expected keywords must appear.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def grade(
        self,
        output: str,
        expected_keywords: Sequence[str],
    ) -> GradeResult:
        """Grade a single response against expected keywords.

        Args:
            output: The agent's text output.
            expected_keywords: Keywords that should appear in a correct response.

        Returns:
            GradeResult with score, matched keywords, and correctness flag.
        """
        if not expected_keywords:
            return GradeResult(
                score=1.0,
                matched_keywords=(),
                total_keywords=0,
                is_correct=True,
            )

        output_lower = output.lower()
        matched = tuple(kw for kw in expected_keywords if kw.lower() in output_lower)
        score = len(matched) / len(expected_keywords)

        return GradeResult(
            score=score,
            matched_keywords=matched,
            total_keywords=len(expected_keywords),
            is_correct=score >= self._threshold,
        )

    def grade_batch(
        self,
        outputs: Sequence[str],
        expected: Sequence[Sequence[str]],
        had_errors: Sequence[bool] | None = None,
    ) -> BatchGradeResult:
        """Grade a batch of responses and detect silent failures.

        Args:
            outputs: Agent output strings.
            expected: Expected keyword lists, one per output.
            had_errors: Whether each run raised an exception. If None,
                assumes no errors (all runs completed without exception).

        Returns:
            BatchGradeResult with accuracy, scores, and silent failure count.
        """
        if len(outputs) != len(expected):
            raise ValueError(
                f"outputs ({len(outputs)}) and expected ({len(expected)}) must have same length"
            )
        if had_errors is not None and len(had_errors) != len(outputs):
            raise ValueError(
                f"had_errors ({len(had_errors)}) must match outputs ({len(outputs)})"
            )

        if had_errors is None:
            had_errors = [False] * len(outputs)

        grades = tuple(self.grade(out, exp) for out, exp in zip(outputs, expected))
        n_total = len(grades)

        if n_total == 0:
            return BatchGradeResult(
                n_total=0,
                n_correct=0,
                n_incorrect=0,
                accuracy=0.0,
                mean_score=0.0,
                silent_failure_count=0,
                silent_failure_rate=0.0,
                grades=(),
            )

        n_correct = sum(1 for g in grades if g.is_correct)
        n_incorrect = n_total - n_correct

        # Silent failure: incorrect response AND no exception was raised
        silent_failures = sum(
            1
            for g, err in zip(grades, had_errors)
            if not g.is_correct and not err
        )

        return BatchGradeResult(
            n_total=n_total,
            n_correct=n_correct,
            n_incorrect=n_incorrect,
            accuracy=n_correct / n_total,
            mean_score=sum(g.score for g in grades) / n_total,
            silent_failure_count=silent_failures,
            silent_failure_rate=silent_failures / n_total,
            grades=grades,
        )
