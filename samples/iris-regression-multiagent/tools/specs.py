"""Tool specification registry for framework-agnostic tool contracts.

Each ToolSpec defines name, description, input_model, output_model, and the
callable — frameworks translate these into their native tool formats (LangChain BaseTool,
ag2 function-calling, CrewAI @tool).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from core.schemas import (
    DataLoadInput,
    DataLoadOutput,
    EvalInput,
    EvalOutput,
    FitInput,
    FitOutput,
    ReportInput,
    ReportOutput,
)
from .functions import (
    load_iris_data_tool,
    fit_linear_regression_tool,
    evaluate_regression_tool,
    write_report_tool,
)


@dataclass(frozen=True)
class ToolSpec:
    """Specification of a single tool (framework-agnostic)."""
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: Callable[[BaseModel], BaseModel]


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "load_iris_data": ToolSpec(
        name="load_iris_data",
        description="Load Iris dataset and perform train/test split with specified features.",
        input_model=DataLoadInput,
        output_model=DataLoadOutput,
        fn=load_iris_data_tool,
    ),
    "fit_linear_regression": ToolSpec(
        name="fit_linear_regression",
        description="Fit a linear regression model on the training data.",
        input_model=FitInput,
        output_model=FitOutput,
        fn=fit_linear_regression_tool,
    ),
    "evaluate_regression": ToolSpec(
        name="evaluate_regression",
        description="Evaluate the fitted model on test data, compute R² and other metrics.",
        input_model=EvalInput,
        output_model=EvalOutput,
        fn=evaluate_regression_tool,
    ),
    "write_report": ToolSpec(
        name="write_report",
        description="Generate a markdown report of the regression results.",
        input_model=ReportInput,
        output_model=ReportOutput,
        fn=write_report_tool,
    ),
}
