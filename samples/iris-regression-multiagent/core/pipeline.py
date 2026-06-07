"""Reference deterministic pipeline (no agents, no LLM) — the ground-truth oracle."""

from .data import clear_cache as clear_data_cache, load_iris_data
from .evaluation import evaluate_regression
from .modeling import clear_cache as clear_model_cache, fit_linear_regression
from .reporting import write_report
from .schemas import (
    DataLoadInput,
    EvalInput,
    FitInput,
    PipelineConfig,
    PipelineRunResult,
    ReportInput,
)


def run_pipeline(config: PipelineConfig | None = None) -> PipelineRunResult:
    """Run a full Iris regression pipeline with optional retraining loop.

    This is the deterministic reference implementation (no agents, no LLM).
    All three frameworks should produce equivalent results.

    Args:
        config: PipelineConfig with thresholds and feature escalation. Defaults to PipelineConfig().

    Returns:
        PipelineRunResult with success, final R², attempt count, and report.
    """
    if config is None:
        config = PipelineConfig()

    # Clear caches at start to ensure clean state
    clear_data_cache()
    clear_model_cache()

    if not config.feature_escalation:
        return PipelineRunResult(
            success=False,
            final_r2=None,
            attempts=0,
            report_markdown=None,
            error="No feature subsets in feature_escalation",
        )

    attempt = 0
    for feature_subset in config.feature_escalation:
        attempt += 1

        try:
            # Load dataset with current feature subset
            data_result = load_iris_data(DataLoadInput(
                feature_subset=feature_subset,
                test_size=0.2,
                random_state=42,
            ))
            dataset_id = data_result.dataset_id

            # Fit model
            fit_result = fit_linear_regression(FitInput(dataset_id=dataset_id))
            model_id = fit_result.model_id

            # Evaluate model
            eval_result = evaluate_regression(EvalInput(model_id=model_id, dataset_id=dataset_id), config)

            # Generate report
            report_result = write_report(ReportInput(model_id=model_id, eval=eval_result, attempt=attempt))

            # Check verdict
            if eval_result.verdict == "accept" or attempt >= config.max_retrain_attempts:
                return PipelineRunResult(
                    success=True,
                    final_r2=eval_result.r2,
                    attempts=attempt,
                    report_markdown=report_result.markdown,
                    error=None,
                )

            # If rejected and we can retry, continue to next feature subset
            if attempt < len(config.feature_escalation):
                continue
            else:
                # Out of feature subsets but still below threshold
                return PipelineRunResult(
                    success=True,
                    final_r2=eval_result.r2,
                    attempts=attempt,
                    report_markdown=report_result.markdown,
                    error=f"Model rejected (R²={eval_result.r2:.4f} < {config.r2_threshold}), no more feature subsets to try",
                )

        except Exception as e:
            return PipelineRunResult(
                success=False,
                final_r2=None,
                attempts=attempt,
                report_markdown=None,
                error=f"Error at attempt {attempt}: {type(e).__name__}: {str(e)}",
            )

    return PipelineRunResult(
        success=False,
        final_r2=None,
        attempts=attempt,
        report_markdown=None,
        error="Pipeline exhausted all feature escalation attempts without acceptance",
    )
