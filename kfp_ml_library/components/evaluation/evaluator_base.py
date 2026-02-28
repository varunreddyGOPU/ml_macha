"""
Evaluator base module.

Provides the abstract ``EvaluatorBase`` class that every
framework-specific evaluator extends.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from kfp_ml_library.configs.evaluator_config import EvaluatorConfig

logger = logging.getLogger(__name__)


class EvaluatorBase(ABC):
    """
    Abstract base class for model evaluation.

    Implementations must override ``_compute_metrics``, ``_generate_plots``
    and ``_compare_with_baseline``.
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config
        self.metrics: Dict[str, float] = {}
        self.plots: Dict[str, Any] = {}
        self.is_blessed: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_metrics(
        self,
        y_true,
        y_pred,
        y_prob=None,
        **kwargs,
    ) -> Dict[str, float]:
        """Compute all relevant evaluation metrics."""
        ...

    @abstractmethod
    def _generate_plots(
        self,
        y_true,
        y_pred,
        y_prob=None,
        feature_names: Optional[List[str]] = None,
        model: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate evaluation plots (confusion matrix, ROC, etc.)."""
        ...

    @abstractmethod
    def _compare_with_baseline(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compare current model with baseline model."""
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true,
        y_pred,
        y_prob=None,
        feature_names: Optional[List[str]] = None,
        model: Any = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full evaluation pipeline: metrics → plots → blessing → optional comparison.

        Returns a comprehensive evaluation report.
        """
        logger.info("Computing metrics…")
        self.metrics = self._compute_metrics(y_true, y_pred, y_prob, **kwargs)

        logger.info("Generating plots…")
        self.plots = self._generate_plots(
            y_true, y_pred, y_prob, feature_names, model, **kwargs
        )

        # Check blessing
        primary = self.config.primary_metric
        primary_value = self.metrics.get(primary, 0.0)
        self.is_blessed = primary_value >= self.config.blessing_threshold

        # Check individual thresholds
        threshold_results: List[Dict] = []
        for t in self.config.thresholds:
            actual = self.metrics.get(t.metric_name, 0.0)
            passed = t.check(actual)
            threshold_results.append({
                "metric": t.metric_name,
                "threshold": t.threshold_value,
                "actual": actual,
                "passed": passed,
            })
            if not passed:
                self.is_blessed = False

        comparison: Dict[str, Any] = {}
        if self.config.compare_with_baseline and baseline_metrics:
            comparison = self._compare_with_baseline(self.metrics, baseline_metrics)

        report = {
            "metrics": self.metrics,
            "threshold_results": threshold_results,
            "is_blessed": self.is_blessed,
            "primary_metric": primary,
            "primary_value": primary_value,
            "comparison": comparison,
        }
        logger.info("Evaluation complete. Blessed=%s", self.is_blessed)
        return report

    def check_blessing(self) -> bool:
        """Return whether the model meets the blessing threshold."""
        return self.is_blessed
