from __future__ import annotations

import logging
from types import TracebackType
from typing import Any, Optional

import mlflow

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Context manager encapsulating an MLflow run for evaluation tracking."""

    def __init__(
        self,
        experiment_name: str,
        run_name: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.params = params or {}
        self._active_run: mlflow.ActiveRun | None = None

    def __enter__(self) -> mlflow.ActiveRun:
        mlflow.set_experiment(self.experiment_name)
        self._active_run = mlflow.start_run(run_name=self.run_name)
        if self.params:
            mlflow.log_params(self.params)
        return self._active_run

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            logger.error("MLflow run '%s' exited with exception: %s", self.run_name, exc_val)
        mlflow.end_run()
        self._active_run = None
        return False

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Logs a flat dictionary of metric names to numeric values."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Logs a local file (report, JSON summary, etc.) as an MLflow artifact."""
        mlflow.log_artifact(local_path, artifact_path=artifact_path)