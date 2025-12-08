"""Base logger protocol."""

from abc import ABC, abstractmethod
from enum import Enum

MISSING_LOGDIR_WANING = "`log_dir` not results might not be saved to disk."


class Status(Enum):
    """Status of a simulator, objective, or observable."""

    STARTED = 0
    RUNNING = 1
    COMPLETE = 2
    ERROR = 3


class StatusKind(Enum):
    """Kind of status for a simulator, objective, or observable."""

    SIMULATOR = 0
    OBJECTIVE = 1
    OBSERVABLE = 2


class Logger(ABC):
    """Base Logger abstract class."""

    @abstractmethod
    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log the `value` for `name` at `step`.

        Args:
            name (str): the name of the metric
            value (float): the value of the metric
            step (int): the step at which the metric was recorded
        """

    @abstractmethod
    def update_status(self, name: str, kind: StatusKind, status: Status) -> None:
        """Updates the status of a simulator, objective, or observable."""

    def update_simulator_status(self, name: str, status: Status) -> None:
        """Updates the status of a simulator."""
        self.update_status(name, StatusKind.SIMULATOR, status)

    def set_simulator_started(self, name: str) -> None:
        """Sets the status of a simulator to STARTED."""
        self.update_simulator_status(name, Status.STARTED)

    def set_simulator_running(self, name: str) -> None:
        """Sets the status of a simulator to RUNNING."""
        self.update_simulator_status(name, Status.RUNNING)

    def set_simulator_complete(self, name: str) -> None:
        """Sets the status of a simulator to COMPLETE."""
        self.update_simulator_status(name, Status.COMPLETE)

    def set_simulator_error(self, name: str) -> None:
        """Sets the status of a simulator to ERROR."""
        self.update_simulator_status(name, Status.ERROR)

    def update_objective_status(self, name: str, status: Status) -> None:
        """Updates the status of an objective."""
        self.update_status(name, StatusKind.OBJECTIVE, status)

    def set_objective_started(self, name: str) -> None:
        """Sets the status of an objective to STARTED."""
        self.update_objective_status(name, Status.STARTED)

    def set_objective_running(self, name: str) -> None:
        """Sets the status of an objective to RUNNING."""
        self.update_objective_status(name, Status.RUNNING)

    def set_objective_complete(self, name: str) -> None:
        """Sets the status of an objective to COMPLETE."""
        self.update_objective_status(name, Status.COMPLETE)

    def set_objective_error(self, name: str) -> None:
        """Sets the status of an objective to ERROR."""
        self.update_objective_status(name, Status.ERROR)

    def update_observable_status(self, name: str, status: Status) -> None:
        """Updates the status of an observable."""
        self.update_status(name, StatusKind.OBSERVABLE, status)

    def set_observable_started(self, name: str) -> None:
        """Sets the status of an observable to STARTED."""
        self.update_observable_status(name, Status.STARTED)

    def set_observable_running(self, name: str) -> None:
        """Sets the status of an observable to RUNNING."""
        self.update_observable_status(name, Status.RUNNING)

    def set_observable_complete(self, name: str) -> None:
        """Sets the status of an observable to COMPLETE."""
        self.update_observable_status(name, Status.COMPLETE)

    def set_observable_error(self, name: str) -> None:
        """Sets the status of an observable to ERROR."""
        self.update_observable_status(name, Status.ERROR)


class NullLogger(Logger):
    """A logger that does nothing."""

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Intentionally Does nothing."""

    def update_status(self, name: str, kind: StatusKind, status: Status) -> None:
        """Intentionally Does nothing."""
