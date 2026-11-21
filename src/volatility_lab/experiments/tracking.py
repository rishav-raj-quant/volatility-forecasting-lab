"""Experiment tracking and metadata persistence for quantitative research."""

import os
import json
import time
import uuid
from typing import Dict, Any, List, Optional
import pandas as pd
from volatility_lab.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Logs and persists research experiment runs for full reproducibility."""

    def __init__(self, log_dir: str = "experiments/results"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.index_file = os.path.join(self.log_dir, "experiments_manifest.jsonl")

    def log_run(
        self,
        ticker: str,
        model_name: str,
        horizon: int,
        metrics: Dict[str, float],
        parameters: Optional[Dict[str, Any]] = None,
        train_start: Optional[str] = None,
        test_end: Optional[str] = None,
        seed: int = 42,
    ) -> str:
        """Records a single model experiment run to the tracking store."""
        exp_id = f"exp_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        record = {
            "experiment_id": exp_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "model_name": model_name,
            "horizon": horizon,
            "metrics": metrics,
            "parameters": parameters or {},
            "train_start": train_start,
            "test_end": test_end,
            "seed": seed,
        }

        with open(self.index_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info(f"Logged experiment {exp_id} for {model_name} (h={horizon}) - QLIKE: {metrics.get('qlike', 0):.4f}")
        return exp_id

    def list_experiments(self) -> pd.DataFrame:
        """Loads and returns all logged experiments as a DataFrame."""
        if not os.path.exists(self.index_file):
            return pd.DataFrame()

        records = []
        with open(self.index_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if not records:
            return pd.DataFrame()

        flat = []
        for r in records:
            row = {
                "experiment_id": r["experiment_id"],
                "timestamp": r["timestamp"],
                "ticker": r["ticker"],
                "model_name": r["model_name"],
                "horizon": r["horizon"],
                "seed": r["seed"],
            }
            for k, v in r["metrics"].items():
                row[f"metric_{k}"] = v
            flat.append(row)

        return pd.DataFrame(flat)
