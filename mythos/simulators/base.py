"""Base class for a simulation."""

import uuid
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, ClassVar

import chex
from typing_extensions import override

import mythos.simulators.io as jd_sio


@chex.dataclass(kw_only=True)
class BaseSimulation(ABC):
    """Base class for a simulation."""

    name: str | None = field(default_factory=uuid.uuid4)
    exposed_observables: ClassVar[list[str]] = ["trajectory"]

    def exposes(self) -> list[str]:
        """Get the list of exposed observables."""
        return [f"{obs}.{self.__class__.__name__}.{self.name}" for obs in self.exposed_observables]

    @abstractmethod
    def run(self, *args, **kwargs) -> jd_sio.SimulatorTrajectory:
        """Run the simulation."""


@chex.dataclass(kw_only=True)
class MultiSimulation(BaseSimulation):
    """A simulation that runs multiple simulations in sequence."""

    simulations: list[BaseSimulation]

    @classmethod
    def create(cls, num: int, sim_class: type[BaseSimulation], /, *sim_args, **sim_kwargs) -> "MultiSimulation":
        """Create a MultiSimulation with n instances of sim_cls."""
        name = sim_kwargs.pop("name", None)
        def name_kwarg(index: int) -> str:
            if name is None:
                return {}
            return {"name": f"{name}.{index}"}
        sims = [sim_class(*sim_args, **sim_kwargs, **name_kwarg(i)) for i in range(num)]
        return cls(simulations=sims)

    @override
    def exposes(self) -> list[str]:
        return [obs for sim in self.simulations for obs in sim.exposes()]

    @override
    def run(self, *args, **kwargs) -> list[Any]:
        outputs = []
        for sim in self.simulations:
            obs = sim.run(*args, **kwargs)
            if len(sim.exposes()) > 1:
                outputs.extend(obs)
            else:
                outputs.append(obs)
        return outputs
