"""Base logger protocol."""

from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from typing_extensions import override

from jax_dna.ui.loggers.logger import Logger, Status, StatusKind

MISSING_LOGDIR_WARNING = "`log_dir` not results might not be saved to disk."

def convert_to_fname(name: str) -> str:
    """Convert a metric name to a valid filename."""
    return name.replace("/", "_").replace(" ", "_") + ".csv"

def tsnow() -> str:
    """Get the current timestamp as a string."""
    return datetime.now(tz=UTC).isoformat()

class FileLogger:
    """Logger that writes all data to a single file."""

    def __init__(self, log_file: str | Path, mode: str = "a") -> "FileLogger":
        self.log_file = Path(log_file).open(mode=mode)

    @override
    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log the `value` for `name` at `step`.

        Args:
            name (str): the name of the metric
            value (float): the value of the metric
            step (int): the step at which the metric was recorded
        """
        self.log_file.write(f"{step},{tsnow()},{name},{value}\n")
        self.log_file.flush()

    @override
    def update_status(self, name: str, kind: StatusKind, status: Status) -> None:
        """Updates the status of a simulator, objective, or observable."""
        self.log_file.write(f"{tsnow()},{name},{status}\n")
        self.log_file.flush()


class PerMetricFileLogger(Logger):
    """Logger that writes each metric/status to its own file."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file_handles = {}

    def _get_file_handle(self, name: str) -> TextIO:
        if name not in self.file_handles:
            fname = self.log_dir / convert_to_fname(name)
            self.file_handles[name] = fname.open(mode="a")
        return self.file_handles[name]

    @override
    def log_metric(self, name: str, value: float, step: int) -> None:
        fh = self._get_file_handle(name)
        fh.write(f"{step},{tsnow()},{value}\n")
        fh.flush()

    @override
    def update_status(self, name: str, kind: StatusKind, status: Status) -> None:
        fh = self._get_file_handle(name)
        fh.write(f"{tsnow()},{status}\n")
        fh.flush()
