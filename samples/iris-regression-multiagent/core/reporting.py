"""Report generation for regression results."""

from __future__ import annotations

from .modeling import get_model
from .schemas import ReportInput, ReportOutput


def write_report(input_config: ReportInput) -> ReportOutput:
    """Generate a markdown report of the regression results.

    Args:
        input_config: ReportInput with model_id, eval results, and attempt number.

    Returns:
        ReportOutput with rendered markdown.

    Raises:
        KeyError: If model_id not found.
    """
    model, feature_names = get_model(input_config.model_id)
    eval_results = input_config.eval

    # Build markdown report
    lines = [
        "# Iris Linear Regression Report",
        "",
        f"## Attempt {input_config.attempt}",
        "",
        "### Model Specification",
        f"- Target: petal width (cm)",
        f"- Features: {', '.join(feature_names)}",
        f"- Algorithm: Linear Regression (sklearn.linear_model.LinearRegression)",
        "",
        "### Coefficients",
    ]

    for fname, coef in zip(feature_names, model.coef_):
        lines.append(f"- {fname}: {coef:.6f}")

    lines.extend([
        f"- Intercept: {model.intercept_:.6f}",
        "",
        "### Test Set Metrics",
        f"- R² Score: {eval_results.r2:.6f}",
        f"- Mean Absolute Error: {eval_results.mae:.6f}",
        f"- Root Mean Squared Error: {eval_results.rmse:.6f}",
        f"- Test Samples: {eval_results.n_samples}",
        "",
        "### Verdict",
        f"- Status: **{eval_results.verdict.upper()}**",
        f"- Threshold: R² >= {input_config.r2_threshold:.2f}",
        f"- Actual R²: {eval_results.r2:.6f}",
    ])

    markdown = "\n".join(lines)
    return ReportOutput(markdown=markdown)
