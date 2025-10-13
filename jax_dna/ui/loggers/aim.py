"""Aim logging backend."""

from aim import Run
from typing_extensions import override

from .logger import Logger, Status


class AimLogger(Logger):
    """Logger that emits metrics and status to Aim."""

    def __init__(self, aim_run: Run = None, **kwargs) -> "AimLogger":
        """Initialize the AimLogger.

        Args:
            aim_run: Aim Run object (optional).
            kwargs: Keyword arguments to initialize a new Aim Run if `aim_run` is not provided.
        """
        if aim_run is None:
            self.aim_run = Run(**kwargs)
        elif kwargs:
            raise ValueError("Cannot provide both an existing aim_run and kwargs.")
        else:
            self.aim_run = aim_run

    @override
    def log_metric(self, name: str, value: float, step: int) -> None:
        value = float(value)  # Give aim python object (iso jax/numpy array obj)
        self.aim_run.track(value, name=name, step=step)

    @override
    def update_status(self, name: str, status: Status) -> None:
        self.aim_run.track(str(status), name=f"status/{name}")
