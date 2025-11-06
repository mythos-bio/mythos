from typing import Any

import chex
import ray
from typing_extensions import override

from mythos.simulators.base import BaseSimulation, MultiSimulation


@ray.remote
class _RaySimulationWrapper:
    def __init__(self, sim_class: type[BaseSimulation], /, *sim_args, **sim_kwargs) -> None:
        self.simulator = sim_class(*sim_args, **sim_kwargs)

    def exposes(self) -> list[str]:
        return self.simulator.exposes()

    def run(self, *args, **kwargs) -> Any:
        return self.simulator.run(*args, **kwargs)

    def call(self, method: str, *args, **kwargs) -> Any:
        func = getattr(self.simulator, method)
        return func(*args, **kwargs)


class RaySimulation(BaseSimulation):
    """A simulation that runs a simulation using Ray."""

    def __init__(self, sim_class: type[BaseSimulation], /, ray_options: dict[str, Any], **sim_kwargs) -> None:
        # create wrapper class
        self.simulator = _RaySimulationWrapper.options(**ray_options).remote(sim_class, **sim_kwargs)

    @override
    def exposes(self) -> list[str]:
        return ray.get(self.simulator.exposes.remote())

    @override
    def run(self, *args, **kwargs) -> Any:
        return ray.get(self.run_async(*args, **kwargs))

    def run_async(self, *args, **kwargs) -> ray.ObjectRef:
        """Runs the simulation asynchronously and returns object references.

        This is the same as run, but does not block and fetch results, but
        rather returns ray.ObjectRef(s) that can be used to fetch results later.
        """
        num_returns = len(self.exposes())
        return self.simulator.run.options(num_returns=num_returns).remote(*args, **kwargs)


@chex.dataclass(kw_only=True)
class RayMultiSimulation(MultiSimulation, RaySimulation):
    """A simulation that runs simulations in parallel using Ray."""

    @override
    @classmethod
    def create(
        cls,
        num: int,
        sim_class: type[BaseSimulation],
        /,
        *sim_args,
        ray_options: dict[str, Any],
        **sim_kwargs
    ) -> "RayMultiSimulation":
        ms = MultiSimulation.create(num, RaySimulation, sim_class, *sim_args, ray_options=ray_options, **sim_kwargs)
        return cls(simulations=ms.simulations)

    @override
    def run(self, *args, **kwargs) -> list[Any]:
        refs = self.run_async(*args, **kwargs)
        return ray.get(refs)

    @override
    def run_async(self, *args, **kwargs) -> list[ray.ObjectRef]:
        refs = []
        for sim in self.simulations:
            sim_refs = sim.run_async(*args, **kwargs)
            if isinstance(sim_refs, ray.ObjectRef):
                refs.append(sim_refs)
            else:
                refs.extend(sim_refs)
        return refs
