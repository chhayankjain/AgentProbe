"""Tests for response grading and silent failure detection."""

from __future__ import annotations

import pytest

from agentprobe.benchmark.grader import (
    BatchGradeResult,
    GradeResult,
    ResponseGrader,
)


class TestGradeResult:
    """Test single response grading."""

    def test_perfect_match(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade(
            "Tokyo has a population of 14 million people",
            ["tokyo", "population", "million"],
        )
        assert result.score == 1.0
        assert result.is_correct is True
        assert len(result.matched_keywords) == 3

    def test_partial_match_above_threshold(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade(
            "The population is large",
            ["tokyo", "population", "million"],
        )
        assert result.score == pytest.approx(1 / 3)
        assert result.is_correct is False

    def test_partial_match_at_threshold(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade(
            "Tokyo population info",
            ["tokyo", "population", "million", "japan"],
        )
        assert result.score == pytest.approx(0.5)
        assert result.is_correct is True

    def test_no_match(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade(
            "I cannot help with that",
            ["tokyo", "population"],
        )
        assert result.score == 0.0
        assert result.is_correct is False
        assert result.matched_keywords == ()

    def test_case_insensitive(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade(
            "TOKYO has a POPULATION of 14 MILLION",
            ["tokyo", "population", "million"],
        )
        assert result.score == 1.0
        assert result.is_correct is True

    def test_empty_keywords(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade("anything", [])
        assert result.score == 1.0
        assert result.is_correct is True
        assert result.total_keywords == 0

    def test_empty_output(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade("", ["tokyo", "population"])
        assert result.score == 0.0
        assert result.is_correct is False

    def test_keyword_substring_match(self):
        """Keywords match as substrings in output."""
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade(
            "The populations of Tokyo and Osaka differ",
            ["population", "tokyo"],
        )
        assert result.score == 1.0

    def test_frozen_result(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade("test", ["test"])
        with pytest.raises(AttributeError):
            result.score = 0.5  # type: ignore[misc]

    def test_total_keywords_recorded(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade("output", ["a", "b", "c", "d", "e"])
        assert result.total_keywords == 5


class TestThreshold:
    """Test threshold configuration."""

    def test_strict_threshold(self):
        grader = ResponseGrader(threshold=1.0)
        result = grader.grade("tokyo population", ["tokyo", "population", "million"])
        assert result.is_correct is False  # 2/3 < 1.0

    def test_lenient_threshold(self):
        grader = ResponseGrader(threshold=0.0)
        result = grader.grade("irrelevant", ["tokyo", "population"])
        assert result.is_correct is True  # 0.0 >= 0.0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            ResponseGrader(threshold=1.5)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError):
            ResponseGrader(threshold=-0.1)


class TestBatchGrading:
    """Test batch grading and aggregation."""

    def test_all_correct(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(
            outputs=["tokyo population 14 million", "berlin capital germany"],
            expected=[["tokyo", "population"], ["berlin", "capital"]],
        )
        assert result.n_total == 2
        assert result.n_correct == 2
        assert result.accuracy == 1.0
        assert result.silent_failure_count == 0

    def test_all_incorrect(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(
            outputs=["unrelated", "nope"],
            expected=[["tokyo", "population"], ["berlin", "capital"]],
        )
        assert result.n_correct == 0
        assert result.accuracy == 0.0

    def test_mixed_results(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(
            outputs=["tokyo population 14 million", "nope"],
            expected=[["tokyo", "population"], ["berlin", "capital"]],
        )
        assert result.n_correct == 1
        assert result.n_incorrect == 1
        assert result.accuracy == 0.5

    def test_empty_batch(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(outputs=[], expected=[])
        assert result.n_total == 0
        assert result.accuracy == 0.0

    def test_mismatched_lengths_raises(self):
        grader = ResponseGrader(threshold=0.5)
        with pytest.raises(ValueError, match="same length"):
            grader.grade_batch(
                outputs=["a", "b"],
                expected=[["x"]],
            )

    def test_grades_tuple_preserved(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(
            outputs=["tokyo", "berlin"],
            expected=[["tokyo"], ["berlin"]],
        )
        assert len(result.grades) == 2
        assert all(isinstance(g, GradeResult) for g in result.grades)


class TestSilentFailureDetection:
    """Test the critical silent failure detection feature."""

    def test_silent_failure_no_error_wrong_answer(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(
            outputs=["Paris is beautiful"],
            expected=[["tokyo", "population"]],
            had_errors=[False],
        )
        assert result.silent_failure_count == 1
        assert result.silent_failure_rate == 1.0

    def test_loud_failure_has_error(self):
        """Incorrect + had error = NOT a silent failure."""
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(
            outputs=["Error occurred"],
            expected=[["tokyo", "population"]],
            had_errors=[True],
        )
        assert result.silent_failure_count == 0

    def test_correct_answer_not_silent_failure(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(
            outputs=["tokyo population 14 million"],
            expected=[["tokyo", "population"]],
            had_errors=[False],
        )
        assert result.silent_failure_count == 0

    def test_mixed_silent_and_loud(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(
            outputs=[
                "wrong answer confidently",  # silent failure (wrong, no error)
                "Error: API timeout",          # loud failure (wrong, had error)
                "tokyo population million",    # correct
            ],
            expected=[
                ["tokyo", "population"],
                ["berlin", "capital"],
                ["tokyo", "population"],
            ],
            had_errors=[False, True, False],
        )
        assert result.silent_failure_count == 1
        assert result.silent_failure_rate == pytest.approx(1 / 3)

    def test_all_silent_failures(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(
            outputs=["wrong", "also wrong", "still wrong"],
            expected=[["a"], ["b"], ["c"]],
            had_errors=[False, False, False],
        )
        assert result.silent_failure_count == 3
        assert result.silent_failure_rate == 1.0

    def test_had_errors_length_mismatch_raises(self):
        grader = ResponseGrader(threshold=0.5)
        with pytest.raises(ValueError, match="must match"):
            grader.grade_batch(
                outputs=["a", "b"],
                expected=[["x"], ["y"]],
                had_errors=[False],
            )

    def test_had_errors_none_defaults_to_no_errors(self):
        grader = ResponseGrader(threshold=0.5)
        result = grader.grade_batch(
            outputs=["wrong"],
            expected=[["expected_keyword"]],
            had_errors=None,
        )
        # No errors assumed → wrong answer is a silent failure
        assert result.silent_failure_count == 1
