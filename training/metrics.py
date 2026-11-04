"""
Training Metrics, Epoch Timing, and TensorBoard Integration.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional
import tensorflow as tf
import pandas as pd

from config import LOG_DIR


class TrainingMetricsTracker:
    """
    Tracks loss histories, timing, and logs metrics to TensorBoard and CSV.
    """

    def __init__(self, log_dir: Path = LOG_DIR, experiment_name: str = "cgan_run"):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = tf.summary.create_file_writer(str(self.log_dir))
        self.history: List[Dict[str, float]] = []

    def log_epoch(self, epoch: int, metrics: Dict[str, float], duration_sec: float) -> None:
        """
        Record scalar metrics to TensorBoard and memory.
        """
        record = {"epoch": epoch, "duration_sec": round(duration_sec, 2)}
        record.update({k: float(v) for k, v in metrics.items()})
        self.history.append(record)

        with self.writer.as_default():
            for name, val in record.items():
                if name != "epoch":
                    tf.summary.scalar(f"metrics/{name}", val, step=epoch)
            tf.summary.scalar("time/epoch_duration_sec", duration_sec, step=epoch)
        self.writer.flush()

    def get_history_df(self) -> pd.DataFrame:
        """Return history as pandas DataFrame."""
        return pd.DataFrame(self.history)

    def save_history_csv(self, output_path: Path) -> None:
        """Export history to CSV."""
        df = self.get_history_df()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable HH:MM:SS or MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"
