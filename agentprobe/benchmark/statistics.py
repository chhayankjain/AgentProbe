"""Statistical analysis module for AgentProbe benchmark results.

Provides confidence intervals, effect sizes, hypothesis tests, and
summary reports for comparing agent framework reliability metrics.

Uses only ``scipy.stats`` and ``numpy`` -- no exotic dependencies.

Usage::

    from agentprobe.benchmark.statistics import (
        proportion_ci,
        mean_ci,
        cohens_h,
        cohens_d,
        compare_proportions,
        compare_distributions,
        compare_multiple_groups,
        build_comparison_report,
        StatisticalReport,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Confidence Intervals
# ---------------------------------------------------------------------------


def proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Parameters
    ----------
    successes:
        Number of successful trials.
    total:
        Total number of trials.
    confidence:
        Confidence level (default 0.95 for a 95% CI).

    Returns
    -------
    (lower, upper) bounds of the confidence interval.

    Raises
    ------
    ValueError
        If *total* is zero or *successes* is outside [0, total].
    """
    if total <= 0:
        raise ValueError("total must be > 0")
    if successes < 0 or successes > total:
        raise ValueError("successes must be in [0, total]")

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    denominator = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(
        p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mean_ci(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Confidence interval for the mean using the t-distribution.

    Parameters
    ----------
    values:
        Sample observations.
    confidence:
        Confidence level.

    Returns
    -------
    (lower, upper) bounds.

    Raises
    ------
    ValueError
        If *values* has fewer than 2 elements.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        raise ValueError("Need at least 2 observations for a CI")
    mean = float(np.mean(arr))
    se = float(stats.sem(arr))
    alpha = 1 - confidence
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return (mean - t_crit * se, mean + t_crit * se)


# ---------------------------------------------------------------------------
# Effect Sizes
# ---------------------------------------------------------------------------

EffectMagnitude = Literal["negligible", "small", "medium", "large"]


def _interpret_d(d: float) -> EffectMagnitude:
    """Interpret Cohen's d magnitude per Cohen (1988)."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _interpret_h(h: float) -> EffectMagnitude:
    """Interpret Cohen's h magnitude per Cohen (1988)."""
    ah = abs(h)
    if ah < 0.2:
        return "negligible"
    if ah < 0.5:
        return "small"
    if ah < 0.8:
        return "medium"
    return "large"


def cohens_h(p1: float, p2: float) -> tuple[float, EffectMagnitude]:
    """Cohen's *h* for comparing two proportions.

    Parameters
    ----------
    p1, p2:
        Proportions in [0, 1].

    Returns
    -------
    (h_value, interpretation) where interpretation is one of
    ``"negligible"``, ``"small"``, ``"medium"``, ``"large"``.
    """
    h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
    return (h, _interpret_h(h))


def cohens_d(
    group1: list[float] | np.ndarray,
    group2: list[float] | np.ndarray,
) -> tuple[float, EffectMagnitude]:
    """Cohen's *d* for comparing two continuous samples (pooled SD).

    Parameters
    ----------
    group1, group2:
        Sample observations.

    Returns
    -------
    (d_value, interpretation).
    """
    a1 = np.asarray(group1, dtype=np.float64)
    a2 = np.asarray(group2, dtype=np.float64)
    n1, n2 = len(a1), len(a2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Both groups need at least 2 observations")
    var1, var2 = float(np.var(a1, ddof=1)), float(np.var(a2, ddof=1))
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        d = 0.0
    else:
        d = (float(np.mean(a1)) - float(np.mean(a2))) / pooled_std
    return (d, _interpret_d(d))


# ---------------------------------------------------------------------------
# Hypothesis Tests
# ---------------------------------------------------------------------------


@dataclass
class HypothesisTestResult:
    """Outcome of a statistical hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool  # at alpha used


def compare_proportions(
    successes1: int,
    total1: int,
    successes2: int,
    total2: int,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Compare two proportions using Fisher's exact test (for small samples)
    or Chi-squared test (otherwise).

    Chooses Fisher's exact when any expected cell count < 5, otherwise
    uses chi-squared with Yates' continuity correction.

    Parameters
    ----------
    successes1, total1:
        Successes and total trials for group 1.
    successes2, total2:
        Successes and total trials for group 2.
    alpha:
        Significance level (default 0.05).
    """
    failures1 = total1 - successes1
    failures2 = total2 - successes2
    table = np.array([[successes1, failures1], [successes2, failures2]])

    # Expected cell counts
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    n = table.sum()
    expected = np.outer(row_sums, col_sums) / n if n > 0 else table

    if np.any(expected < 5):
        # Fisher's exact test
        odds_ratio, p = stats.fisher_exact(table)
        return HypothesisTestResult(
            test_name="Fisher's exact test",
            statistic=odds_ratio,
            p_value=float(p),
            significant=float(p) < alpha,
        )
    else:
        chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
        return HypothesisTestResult(
            test_name="Chi-squared test (Yates)",
            statistic=float(chi2),
            p_value=float(p),
            significant=float(p) < alpha,
        )


def compare_distributions(
    group1: list[float] | np.ndarray,
    group2: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Mann-Whitney U test for comparing two continuous distributions.

    Non-parametric -- does not assume normality.
    """
    a1 = np.asarray(group1, dtype=np.float64)
    a2 = np.asarray(group2, dtype=np.float64)
    if len(a1) < 1 or len(a2) < 1:
        raise ValueError("Both groups need at least 1 observation")
    stat, p = stats.mannwhitneyu(a1, a2, alternative="two-sided")
    return HypothesisTestResult(
        test_name="Mann-Whitney U test",
        statistic=float(stat),
        p_value=float(p),
        significant=float(p) < alpha,
    )


def compare_multiple_groups(
    groups: list[list[float] | np.ndarray],
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Kruskal-Wallis H-test for comparing 3+ independent groups.

    Non-parametric -- suitable for latency comparisons across frameworks.
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups")
    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    stat, p = stats.kruskal(*arrays)
    return HypothesisTestResult(
        test_name="Kruskal-Wallis H test",
        statistic=float(stat),
        p_value=float(p),
        significant=float(p) < alpha,
    )


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------


@dataclass
class PairwiseComparison:
    """Statistical comparison between two frameworks."""

    framework_a: str
    framework_b: str

    # Proportion (failure rate) comparison
    failure_rate_a: float
    failure_rate_b: float
    failure_rate_ci_a: tuple[float, float]
    failure_rate_ci_b: tuple[float, float]
    failure_rate_test: HypothesisTestResult
    failure_rate_effect: tuple[float, EffectMagnitude]

    # Latency comparison
    latency_test: HypothesisTestResult | None = None
    latency_effect: tuple[float, EffectMagnitude] | None = None


@dataclass
class StatisticalReport:
    """Full statistical comparison report across frameworks.

    Attributes
    ----------
    pairwise:
        List of PairwiseComparison for each framework pair.
    omnibus_latency:
        Kruskal-Wallis test across all frameworks (if 3+).
    n_frameworks:
        Number of frameworks compared.
    alpha:
        Significance level used for all tests.
    """

    pairwise: list[PairwiseComparison] = field(default_factory=list)
    omnibus_latency: HypothesisTestResult | None = None
    n_frameworks: int = 0
    alpha: float = 0.05


@dataclass
class FrameworkData:
    """Input data for a single framework to build_comparison_report.

    Attributes
    ----------
    name:
        Framework identifier (e.g. ``"langgraph"``).
    successes:
        Number of successful runs.
    total:
        Total runs.
    latencies:
        Per-run latency values in milliseconds.
    """

    name: str
    successes: int
    total: int
    latencies: list[float] | np.ndarray


def build_comparison_report(
    frameworks: list[FrameworkData],
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> StatisticalReport:
    """Produce a full pairwise statistical comparison across frameworks.

    Parameters
    ----------
    frameworks:
        One FrameworkData per framework.
    alpha:
        Significance level for hypothesis tests.
    confidence:
        Confidence level for CIs.

    Returns
    -------
    A StatisticalReport with pairwise comparisons and an omnibus
    latency test (when 3+ frameworks are provided).
    """
    report = StatisticalReport(n_frameworks=len(frameworks), alpha=alpha)

    # Pairwise comparisons
    for i in range(len(frameworks)):
        for j in range(i + 1, len(frameworks)):
            fa, fb = frameworks[i], frameworks[j]

            rate_a = (fa.total - fa.successes) / fa.total if fa.total else 0.0
            rate_b = (fb.total - fb.successes) / fb.total if fb.total else 0.0
            ci_a = proportion_ci(fa.total - fa.successes, fa.total, confidence)
            ci_b = proportion_ci(fb.total - fb.successes, fb.total, confidence)

            prop_test = compare_proportions(
                fa.total - fa.successes,
                fa.total,
                fb.total - fb.successes,
                fb.total,
                alpha=alpha,
            )
            h_val, h_interp = cohens_h(rate_a, rate_b)

            lat_arr_a = np.asarray(fa.latencies, dtype=np.float64)
            lat_arr_b = np.asarray(fb.latencies, dtype=np.float64)

            lat_test = None
            lat_effect = None
            if len(lat_arr_a) >= 2 and len(lat_arr_b) >= 2:
                lat_test = compare_distributions(lat_arr_a, lat_arr_b, alpha=alpha)
                lat_effect = cohens_d(lat_arr_a, lat_arr_b)

            report.pairwise.append(
                PairwiseComparison(
                    framework_a=fa.name,
                    framework_b=fb.name,
                    failure_rate_a=rate_a,
                    failure_rate_b=rate_b,
                    failure_rate_ci_a=ci_a,
                    failure_rate_ci_b=ci_b,
                    failure_rate_test=prop_test,
                    failure_rate_effect=(h_val, h_interp),
                    latency_test=lat_test,
                    latency_effect=lat_effect,
                )
            )

    # Omnibus latency test (3+ frameworks)
    if len(frameworks) >= 3:
        lat_groups = [
            np.asarray(f.latencies, dtype=np.float64)
            for f in frameworks
            if len(np.asarray(f.latencies)) >= 2
        ]
        if len(lat_groups) >= 3:
            report.omnibus_latency = compare_multiple_groups(
                lat_groups, alpha=alpha
            )

    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "proportion_ci",
    "mean_ci",
    "cohens_h",
    "cohens_d",
    "compare_proportions",
    "compare_distributions",
    "compare_multiple_groups",
    "build_comparison_report",
    "EffectMagnitude",
    "HypothesisTestResult",
    "PairwiseComparison",
    "StatisticalReport",
    "FrameworkData",
]
