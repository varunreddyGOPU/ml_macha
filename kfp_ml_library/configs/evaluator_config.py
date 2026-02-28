"""
Evaluator configuration module.

Defines thresholds, metric selections, and comparison strategies
for model evaluation components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kfp_ml_library.configs.constants import (
    DEFAULT_EVALUATION_METRICS,
    DEFAULT_REGRESSION_METRICS,
    MODEL_PERFORMANCE_THRESHOLD,
    TaskType,
)


@dataclass
class EvaluationThreshold:
    """A single metric threshold."""

    metric_name: str
    threshold_value: float
    comparison: str = "gte"  # gte | lte | gt | lt | eq

    def check(self, actual_value: float) -> bool:
        ops = {
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "eq": lambda a, b: abs(a - b) < 1e-9,
        }
        return ops.get(self.comparison, ops["gte"])(actual_value, self.threshold_value)


@dataclass
class SlicingConfig:
    """Configuration for sliced evaluation (evaluation on data subsets)."""

    slicing_column: str = ""
    slicing_values: List[str] = field(default_factory=list)
    min_samples_per_slice: int = 100


@dataclass
class EvaluatorConfig:
    """
    Full evaluation configuration.

    Example::

        config = EvaluatorConfig(
            task_type=TaskType.CLASSIFICATION,
            primary_metric="f1_score",
            thresholds=[
                EvaluationThreshold("accuracy", 0.90),
                EvaluationThreshold("f1_score", 0.85),
            ],
        )
    """

    task_type: TaskType = TaskType.CLASSIFICATION
    primary_metric: str = "accuracy"
    metrics: List[str] = field(default_factory=lambda: list(DEFAULT_EVALUATION_METRICS))
    regression_metrics: List[str] = field(
        default_factory=lambda: list(DEFAULT_REGRESSION_METRICS)
    )
    thresholds: List[EvaluationThreshold] = field(default_factory=list)
    global_threshold: float = MODEL_PERFORMANCE_THRESHOLD
    compare_with_baseline: bool = True
    baseline_model_path: Optional[str] = None
    slicing_configs: List[SlicingConfig] = field(default_factory=list)
    generate_confusion_matrix: bool = True
    generate_roc_curve: bool = True
    generate_feature_importance: bool = True
    output_report_path: str = "/tmp/evaluation_report"
    blessing_threshold: float = MODEL_PERFORMANCE_THRESHOLD

    def get_metrics_for_task(self) -> List[str]:
        """Return the appropriate metrics list based on the task type."""
        if self.task_type in (TaskType.REGRESSION, TaskType.TIME_SERIES):
            return self.regression_metrics
        return self.metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value if isinstance(self.task_type, TaskType) else self.task_type,
            "primary_metric": self.primary_metric,
            "metrics": self.get_metrics_for_task(),
            "thresholds": [
                {"metric": t.metric_name, "value": t.threshold_value, "comparison": t.comparison}
                for t in self.thresholds
            ],
            "global_threshold": self.global_threshold,
            "compare_with_baseline": self.compare_with_baseline,
            "baseline_model_path": self.baseline_model_path,
            "generate_confusion_matrix": self.generate_confusion_matrix,
            "generate_roc_curve": self.generate_roc_curve,
            "generate_feature_importance": self.generate_feature_importance,
            "blessing_threshold": self.blessing_threshold,
        }
