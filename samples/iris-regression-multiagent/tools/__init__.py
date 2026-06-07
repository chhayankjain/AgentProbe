"""Tools layer: framework-agnostic tool contracts and implementations."""

from .functions import (
    evaluate_regression_tool,
    fit_linear_regression_tool,
    load_iris_data_tool,
    write_report_tool,
)
from .specs import TOOL_REGISTRY, ToolSpec

__all__ = [
    "load_iris_data_tool",
    "fit_linear_regression_tool",
    "evaluate_regression_tool",
    "write_report_tool",
    "TOOL_REGISTRY",
    "ToolSpec",
]
