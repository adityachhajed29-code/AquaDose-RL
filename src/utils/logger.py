"""Telemetry Logger for chemical dosing and reactor state observation.

Tracks time-series logs of pH, control actions, chemical consumption, and rewards.
Provides exports to pandas DataFrames and CSV files for evaluation analysis.
"""

import os
from typing import Any, Dict, List, Optional
import pandas as pd


class TelemetryLogger:
    """In-memory and file-based telemetry logger for CSTR operations."""

    def __init__(self, experiment_name: str = "run", log_dir: Optional[str] = "logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.records: List[Dict[str, Any]] = []

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def log_step(
        self,
        episode: int,
        step: int,
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Record telemetry for a single environment step."""
        entry = {
            "episode": episode,
            "step": step,
            "reward": reward,
            "ph": info.get("ph", 7.0),
            "target_ph": info.get("target_ph", 7.0),
            "error": info.get("error", 0.0),
            "action": info.get("action", 0.0),
            "acid_flow": info.get("acid_flow", 0.0),
            "base_flow": info.get("base_flow", 0.0),
            "reagent_step": info.get("reagent_step", 0.0),
            "cumulative_reagent": info.get("cumulative_reagent", 0.0),
            "flow_in": info.get("flow_in", 0.0),
            "influent_ph": info.get("influent_ph", 0.0),
        }
        self.records.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected telemetry records into a pandas DataFrame."""
        return pd.DataFrame(self.records)

    def save_csv(self, filename: Optional[str] = None) -> str:
        """Save telemetry log to CSV file."""
        if filename is None:
            filename = f"{self.experiment_name}_telemetry.csv"

        target_path = os.path.join(self.log_dir or ".", filename)
        df = self.to_dataframe()
        df.to_csv(target_path, index=False)
        return target_path

    def clear(self) -> None:
        """Clear recorded telemetry."""
        self.records.clear()
