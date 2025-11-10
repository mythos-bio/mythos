"""Base class for a simulation."""

import uuid
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import chex
from typing_extensions import override

import mythos.simulators.io as jd_sio


@chex.dataclass(kw_only=True, eq=False)
class BaseSimulation(ABC):
    """Base class for a simulation."""

    name: str | None = None
    exposed_observables: ClassVar[list[str]] = ["trajectory"]

    @override
    def __post_init__(self) -> None:
        self.name = str(uuid.uuid4()) if self.name is None else self.name

    def exposes(self) -> list[str]:
        """Get the list of exposed observables."""
        return [f"{obs}.{self.__class__.__name__}.{self.name}" for obs in self.exposed_observables]

    @abstractmethod
    def run(self, *args, **kwargs) -> jd_sio.SimulatorTrajectory:
        """Run the simulation."""

    @override
    def __hash__(self) -> int:
        return object.__hash__(self)


@chex.dataclass(kw_only=True)
class MultiSimulation(BaseSimulation):
    """A simulation that runs multiple simulations in sequence."""

    simulations: list[BaseSimulation]

    @classmethod
    def create(cls, num: int, sim_class: type[BaseSimulation], /, *sim_args, **sim_kwargs) -> "MultiSimulation":
        """Create a MultiSimulation with n instances of sim_cls."""
        base_name = sim_kwargs.pop("name", str(uuid.uuid4()))
        sims = [sim_class(*sim_args, **sim_kwargs, name=f"{base_name}.{i}") for i in range(num)]
        return cls(simulations=sims, name=base_name)

    @override
    def exposes(self) -> list[str]:
        return [obs for sim in self.simulations for obs in sim.exposes()]

    @override
    def run(self, *args, **kwargs) -> list[Any]:
        outputs = []
        for sim in self.simulations:
            obs = sim.run(*args, **kwargs)
            obs = [obs] if len(sim.exposes()) == 1 else obs
            outputs.extend(obs)
        return outputs


@chex.dataclass(kw_only=True, eq=False)
class AsyncSimulation(BaseSimulation):
    """An abstract base class for asynchronous simulations.

    This class extends BaseSimulation to provide an interface for running
    simulations asynchronously. Typically, concrete implementations would
    implement the `run_async` method to return future-like objects (e.g.,
    `ray.ObjectRef`s) that can be used to fetch results later. The `run` method
    should typically be implemented to call `run_async` and block and fetch
    results immediately, matching the synchronous api of BaseSimulation.
    """

    @abstractmethod
    def run_async(self, *args, **kwargs) -> Any:
        """Runs the simulation asynchronously and returns future-like objects."""
