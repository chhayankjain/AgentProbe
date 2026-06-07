"""Core deterministic ML pipeline (framework-agnostic)."""

from .data import clear_cache as clear_data_cache, load_iris_data
from .evaluation import evaluate_regression
from .modeling import clear_cache as clear_model_cache, fit_linear_regression
from .pipeline import run_pipeline
from .reporting import write_report
from .schemas import (
    DataLoadInput,
    DataLoadOutput,
    EvalInput,
    EvalOutput,
    FitInput,
    FitOutput,
    PipelineConfig,
    PipelineRunResult,
    ReportInput,
    ReportOutput,
)

__all__ = [
    "load_iris_data",
    "fit_linear_regression",
    "evaluate_regression",
    "write_report",
    "run_pipeline",
    "clear_data_cache",
    "clear_model_cache",
    "DataLoadInput",
    "DataLoadOutput",
    "FitInput",
    "FitOutput",
    "EvalInput",
    "EvalOutput",
    "ReportInput",
    "ReportOutput",
    "PipelineConfig",
    "PipelineRunResult",
]
