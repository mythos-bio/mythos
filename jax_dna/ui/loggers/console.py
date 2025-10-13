"""Logger logging jax_dna optimization results to the console."""

from typing_extensions import override

from jax_dna.ui.loggers.logger import Logger, Status, StatusKind


class ConsoleLogger(Logger):
    """Console logger."""

    @override
    def log_metric(self, name: str, value: float, step: int) -> None:
        print(f"Step: {step}, {name}: {value}")  # noqa: T201 -- we intend to print to the console

    @override
    def update_status(self, name: str, kind: StatusKind, status: Status) -> None:
        print(name, status)  # noqa: T201 -- we intend to print to the console
