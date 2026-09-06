"""Tests for agentprobe.benchmark.statistics module.

Covers confidence intervals, effect sizes, hypothesis tests, edge cases,
and the summary report builder.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agentprobe.benchmark.statistics import (
    FrameworkData,
    StatisticalReport,
    HypothesisTestResult,
    build_comparison_report,
    cohens_d,
    cohens_h,
    compare_distributions,
    compare_multiple_groups,
    compare_proportions,
    mean_ci,
    proportion_ci,
)


# ===================================================================
# proportion_ci
# ===================================================================


class TestProportionCI:
    """Wilson score interval for proportions."""

    def test_50_percent_rate(self):
        """50/100 at 95% should give roughly [0.40, 0.60]."""
        lo, hi = proportion_ci(50, 100, confidence=0.95)
        assert 0.39 < lo < 0.42
        assert 0.58 < hi < 0.61

    def test_all_successes(self):
        """100/100 should NOT give (1.0, 1.0) -- Wilson shrinks towards 0.5."""
        lo, hi = proportion_ci(100, 100)
        assert lo < 1.0
        assert hi == 1.0  # upper bound clamped

    def test_all_failures(self):
        """0/100 should NOT give (0.0, 0.0)."""
        lo, hi = proportion_ci(0, 100)
        assert lo < 0.001  # lower bound near zero (Wilson score may not be exactly 0)
        assert hi > 0.0

    def test_small_sample(self):
        """3/5 -- CI should be wide."""
        lo, hi = proportion_ci(3, 5)
        assert lo < 0.3
        assert hi > 0.7

    def test_higher_confidence_wider(self):
        """99% CI should be wider than 90% CI."""
        lo90, hi90 = proportion_ci(50, 100, confidence=0.90)
        lo99, hi99 = proportion_ci(50, 100, confidence=0.99)
        assert (hi99 - lo99) > (hi90 - lo90)

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="total must be > 0"):
            proportion_ci(0, 0)

    def test_negative_successes_raises(self):
        with pytest.raises(ValueError, match="successes must be in"):
            proportion_ci(-1, 10)

    def test_successes_exceeds_total_raises(self):
        with pytest.raises(ValueError, match="successes must be in"):
            proportion_ci(11, 10)


# ===================================================================
# mean_ci
# ===================================================================


class TestMeanCI:
    """T-distribution confidence interval for the mean."""

    def test_known_values(self):
        """Constant values should give a zero-width CI."""
        lo, hi = mean_ci([5.0, 5.0, 5.0, 5.0])
        assert abs(lo - 5.0) < 1e-10
        assert abs(hi - 5.0) < 1e-10

    def test_contains_true_mean(self):
        """For a large sample from N(100, 10), CI should contain 100."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, size=500).tolist()
        lo, hi = mean_ci(data, confidence=0.99)
        assert lo < 100 < hi

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="at least 2"):
            mean_ci([42.0])

    def test_wider_at_higher_confidence(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo90, hi90 = mean_ci(data, confidence=0.90)
        lo99, hi99 = mean_ci(data, confidence=0.99)
        assert (hi99 - lo99) > (hi90 - lo90)


# ===================================================================
# cohens_h
# ===================================================================


class TestCohensH:
    def test_identical_proportions(self):
        h, interp = cohens_h(0.5, 0.5)
        assert abs(h) < 1e-10
        assert interp == "negligible"

    def test_large_difference(self):
        """0.1 vs 0.9 should be a large effect."""
        h, interp = cohens_h(0.9, 0.1)
        assert interp == "large"
        assert h > 0  # p1 > p2 means positive h

    def test_symmetry(self):
        h1, _ = cohens_h(0.3, 0.6)
        h2, _ = cohens_h(0.6, 0.3)
        assert abs(h1 + h2) < 1e-10

    def test_hand_computed(self):
        """h = 2*(arcsin(sqrt(0.75)) - arcsin(sqrt(0.25)))"""
        expected = 2 * (math.asin(math.sqrt(0.75)) - math.asin(math.sqrt(0.25)))
        h, _ = cohens_h(0.75, 0.25)
        assert abs(h - expected) < 1e-10


# ===================================================================
# cohens_d
# ===================================================================


class TestCohensD:
    def test_identical_groups(self):
        d, interp = cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(d) < 1e-10
        assert interp == "negligible"

    def test_large_difference(self):
        g1 = [100.0] * 30
        g2 = [0.0] * 30
        # With one group all 100 and other all 0, pooled SD = 0 edge case
        # Use groups with variance instead
        rng = np.random.default_rng(123)
        g1 = (rng.normal(100, 5, 50)).tolist()
        g2 = (rng.normal(0, 5, 50)).tolist()
        d, interp = cohens_d(g1, g2)
        assert interp == "large"
        assert d > 0

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            cohens_d([1.0], [1.0, 2.0])

    def test_zero_variance(self):
        """All identical values in both groups -> d = 0."""
        d, interp = cohens_d([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert d == 0.0
        assert interp == "negligible"


# ===================================================================
# compare_proportions
# ===================================================================


class TestCompareProportions:
    def test_identical_proportions_not_significant(self):
        result = compare_proportions(50, 100, 50, 100)
        assert not result.significant
        assert result.p_value > 0.05

    def test_very_different_proportions_significant(self):
        result = compare_proportions(90, 100, 10, 100)
        assert result.significant
        assert result.p_value < 0.001

    def test_small_sample_uses_fisher(self):
        """With small expected counts, should use Fisher's exact."""
        result = compare_proportions(2, 5, 0, 5)
        assert result.test_name == "Fisher's exact test"

    def test_large_sample_uses_chi2(self):
        result = compare_proportions(80, 200, 90, 200)
        assert "Chi-squared" in result.test_name

    def test_returns_test_result(self):
        result = compare_proportions(50, 100, 60, 100)
        assert isinstance(result, HypothesisTestResult)
        assert isinstance(result.p_value, float)
        assert isinstance(result.statistic, float)


# ===================================================================
# compare_distributions
# ===================================================================


class TestCompareDistributions:
    def test_identical_not_significant(self):
        rng = np.random.default_rng(7)
        data = rng.normal(50, 10, 100)
        result = compare_distributions(data, data)
        assert not result.significant

    def test_different_significant(self):
        rng = np.random.default_rng(7)
        g1 = rng.normal(50, 5, 100).tolist()
        g2 = rng.normal(100, 5, 100).tolist()
        result = compare_distributions(g1, g2)
        assert result.significant
        assert result.test_name == "Mann-Whitney U test"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            compare_distributions([], [1.0])


# ===================================================================
# compare_multiple_groups
# ===================================================================


class TestCompareMultipleGroups:
    def test_identical_groups_not_significant(self):
        rng = np.random.default_rng(99)
        g = rng.normal(50, 10, 50).tolist()
        result = compare_multiple_groups([g, g, g])
        assert not result.significant

    def test_different_groups_significant(self):
        rng = np.random.default_rng(99)
        g1 = rng.normal(10, 2, 50).tolist()
        g2 = rng.normal(50, 2, 50).tolist()
        g3 = rng.normal(90, 2, 50).tolist()
        result = compare_multiple_groups([g1, g2, g3])
        assert result.significant
        assert result.test_name == "Kruskal-Wallis H test"

    def test_fewer_than_two_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compare_multiple_groups([[1.0, 2.0]])


# ===================================================================
# build_comparison_report
# ===================================================================


class TestBuildComparisonReport:
    def _make_frameworks(self) -> list[FrameworkData]:
        rng = np.random.default_rng(42)
        return [
            FrameworkData(
                name="langgraph",
                successes=80,
                total=100,
                latencies=rng.normal(200, 30, 100).tolist(),
            ),
            FrameworkData(
                name="langchain",
                successes=60,
                total=100,
                latencies=rng.normal(300, 50, 100).tolist(),
            ),
            FrameworkData(
                name="autogen",
                successes=70,
                total=100,
                latencies=rng.normal(500, 80, 100).tolist(),
            ),
        ]

    def test_report_structure(self):
        fws = self._make_frameworks()
        report = build_comparison_report(fws)
        assert isinstance(report, StatisticalReport)
        assert report.n_frameworks == 3
        # 3 frameworks -> C(3,2) = 3 pairwise comparisons
        assert len(report.pairwise) == 3

    def test_omnibus_present_for_three(self):
        fws = self._make_frameworks()
        report = build_comparison_report(fws)
        assert report.omnibus_latency is not None
        assert report.omnibus_latency.test_name == "Kruskal-Wallis H test"

    def test_no_omnibus_for_two(self):
        fws = self._make_frameworks()[:2]
        report = build_comparison_report(fws)
        assert report.omnibus_latency is None
        assert len(report.pairwise) == 1

    def test_pairwise_has_effect_sizes(self):
        fws = self._make_frameworks()
        report = build_comparison_report(fws)
        for pw in report.pairwise:
            assert pw.failure_rate_effect is not None
            assert pw.latency_effect is not None
            # Effect magnitude should be a valid string
            assert pw.failure_rate_effect[1] in (
                "negligible", "small", "medium", "large"
            )

    def test_pairwise_failure_rates_correct(self):
        fws = self._make_frameworks()
        report = build_comparison_report(fws)
        pw = report.pairwise[0]  # langgraph vs langchain
        # langgraph: 20% failure, langchain: 40% failure
        assert abs(pw.failure_rate_a - 0.20) < 1e-10
        assert abs(pw.failure_rate_b - 0.40) < 1e-10

    def test_ci_bounds_ordered(self):
        fws = self._make_frameworks()
        report = build_comparison_report(fws)
        for pw in report.pairwise:
            lo_a, hi_a = pw.failure_rate_ci_a
            lo_b, hi_b = pw.failure_rate_ci_b
            assert lo_a <= hi_a
            assert lo_b <= hi_b
