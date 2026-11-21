"""Script to execute walk-forward volatility forecasting experiments."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from volatility_lab.data.ingestion import DataIngestionPipeline
from volatility_lab.features.feature_pipeline import FeaturePipeline
from volatility_lab.machine_learning.registry import MLModelRegistry
from volatility_lab.evaluation.walk_forward import WalkForwardValidator
from volatility_lab.experiments.tracking import ExperimentTracker
from volatility_lab.utils.logging import get_logger

logger = get_logger("run_experiment")


def main():
    ticker = "^NSEI"
    horizon = 5

    logger.info(f"Running walk-forward experiment on {ticker} for horizon {horizon} days...")
    pipeline = DataIngestionPipeline()
    df = pipeline.run(ticker=ticker)

    f_pipe = FeaturePipeline()
    X, y = f_pipe.transform(df, horizon=horizon)

    validator = WalkForwardValidator(min_train_size=500, test_size=126, step_size=126)
    tracker = ExperimentTracker()

    for model_name in ["ridge", "random_forest", "lightgbm"]:
        logger.info(f"Evaluating {model_name}...")
        res = validator.evaluate_model(
            model_factory=lambda m=model_name: MLModelRegistry.get(m),
            X=X,
            y=y,
            forecast_horizon=horizon,
        )

        exp_id = tracker.log_run(
            ticker=ticker,
            model_name=model_name,
            horizon=horizon,
            metrics=res["overall_metrics"],
            train_start=str(X.index[0].date()),
            test_end=str(X.index[-1].date()),
        )
        logger.info(f"Experiment {exp_id} logged. Overall QLIKE: {res['overall_metrics']['qlike']:.4f}")


if __name__ == "__main__":
    main()
