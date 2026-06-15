"""Tool function implementations: validate input, delegate to core/, return output."""

from __future__ import annotations

from core.data import load_iris_data as core_load_iris_data
from core.evaluation import evaluate_regression as core_evaluate_regression
from core.modeling import fit_linear_regression as core_fit_linear_regression
from core.reporting import write_report as core_write_report
from core.schemas import (
    DataLoadInput,
    DataLoadOutput,
    EvalInput,
    EvalOutput,
    FitInput,
    FitOutput,
    PipelineConfig,
    ReportInput,
    ReportOutput,
)


def load_iris_data_tool(input_config: DataLoadInput) -> DataLoadOutput:
    """Load Iris data tool (validates input via Pydantic, delegates to core/)."""
    return core_load_iris_data(input_config)


def fit_linear_regression_tool(input_config: FitInput) -> FitOutput:
    """Fit linear regression tool (validates input via Pydantic, delegates to core/)."""
    return core_fit_linear_regression(input_config)


def evaluate_regression_tool(input_config: EvalInput) -> EvalOutput:
    """Evaluate regression tool (validates input via Pydantic, delegates to core/).

    Uses default PipelineConfig for threshold.
    """
    config = PipelineConfig()  # Default thresholds
    return core_evaluate_regression(input_config, config)


def write_report_tool(input_config: ReportInput) -> ReportOutput:
    """Write report tool (validates input via Pydantic, delegates to core/)."""
    return core_write_report(input_config)
