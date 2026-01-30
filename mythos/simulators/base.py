"""Base class for a simulation."""

import shutil
import uuid
from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, ClassVar

import chex
from typing_extensions import override

from mythos.utils.scheduler import SchedulerUnit


@chex.dataclass(frozen=True)
class SimulatorOutput:
    """Output container for simlators."""
    observables: list[Any]
    state: dict[str, Any] = field(default_factory=dict)


@chex.dataclass(frozen=True, kw_only=True)
class Simulator(SchedulerUnit):
    """Base class for a simulation."""
    name: str = field(default_factory=lambda: str(uuid.uuid4()))
    exposed_observables: ClassVar[list[str]] = ["trajectory"]

    def run(self, *_args, opt_params: dict[str, Any], **_kwargs) -> SimulatorOutput:
        """Run the simulation."""

    def exposes(self) -> list[str]:
        """Get the list of exposed observables."""
        return [f"{obs}.{self.__class__.__name__}.{self.name}" for obs in self.exposed_observables]

    @classmethod
    def create_n(cls, n: int, name: str|None = None, **kwargs) -> list["Simulator"]:
        """Create N simulators with unique names."""
        name = name or str(uuid.uuid4())
        return [cls(name=f"{name}.{i}", **kwargs) for i in range(n)]


@chex.dataclass(frozen=True, kw_only=True)
class InputDirSimulator(Simulator, ABC):
    """A base class for simulators that run based on an input directory.

    This class handles copying the input directory to a temporary location
    unless overwrite_input is set to True.

    Subclasses must implement the run_simulation method, which runs the
    simulation logic given the provided input directory.

    Arguments:
        input_dir: Path to the input directory.
        overwrite_input: Whether to overwrite the input directory or copy it. If
            this is False (default), the contents of the input_dir will be
            copied to a temporary directory for running the simulation to avoid
            overwriting input.
    """

    input_dir: str
    overwrite_input: bool = False

    @override
    def run(self, *args, **kwargs) -> SimulatorOutput:
        if self.overwrite_input:
            return self.run_simulation(Path(self.input_dir), *args, **kwargs)

        with TemporaryDirectory(prefix=f"mythos-sim-{self.name}") as temp_dir:
            self.copy_inputs(temp_dir)
            return self.run_simulation(Path(temp_dir), *args, **kwargs)

    def copy_inputs(self, temp_dir: str) -> None:
        """Copy input files to temporary directory."""
        shutil.copytree(self.input_dir, temp_dir, dirs_exist_ok=True)

    @abstractmethod
    def run_simulation(self, input_dir: Path, *args, **kwargs) -> SimulatorOutput:
        """Run the simulation in the given input directory."""
