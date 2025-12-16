"""Runs an optimization loop using Ray actors for objectives and simulators."""

import dataclasses as dc
import typing
from abc import ABC, abstractmethod

import chex
import optax
import ray
from typing_extensions import override

import mythos.optimization.objective as jdna_objective
import mythos.utils.types as jdna_types
from mythos.simulators.base import AsyncSimulation, BaseSimulation
from mythos.ui.loggers import logger as jdna_logger

ERR_MISSING_OBJECTIVES = "At least one objective is required."
ERR_MISSING_SIMULATORS = "At least one simulator is required."
ERR_MISSING_AGG_GRAD_FN = "An aggregate gradient function is required."
ERR_MISSING_OPTIMIZER = "An optimizer is required."

OptResult = tuple[optax.OptState, jdna_types.Params, jdna_types.Grads]


class Optimizer(ABC):
    """An abstract base class for optimizers.

    Optimizers are a class of objects that take one or more objectives which in
    turn require one or more observables produced via simulators. The optimizer
    coordinates passing information between these components in steps and
    implements an optimization loop.
    """


    @abstractmethod
    def step(self, params: jdna_types.Params) -> OptResult:
        """Perform a single optimization step.

        Args:
            params: Parameters to optimize in the current step.

        Returns:
            A tuple containing the updated optimizer state, new params, and the gradients.
        """

    @abstractmethod
    def post_step(self, optimizer_state: optax.OptState, opt_params: jdna_types.Params) -> "Optimizer":
        """An update step intended to be called after an optimization step.

        Args:
            optimizer_state: The state of the optimizer after the step.
            opt_params: The optimized parameters after the step.

        Returns:
            An updated instance of the Optimizer.
        """

    def get_updates_and_state(self, grads: jdna_types.Grads, params: jdna_types.Params) -> OptResult:
        """Helper function to get the updated optimizer state and new params."""
        opt_state = self.optimizer.init(params) if self.optimizer_state is None else self.optimizer_state
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return opt_state, new_params, grads

    @staticmethod
    def optimize(optimizer: "Optimizer", initial_params: jdna_types.Params, n_steps: int) -> jdna_types.Params:
        """Run the optimization loop for a given number of steps.

        Args:
            optimizer: The optimizer to use for the optimization.
            initial_params: The initial parameters to start optimization from.
            n_steps: The number of optimization steps to perform.

        Returns:
            The optimized parameters after the specified number of steps.
        """
        params = initial_params
        for _ in range(n_steps):
            opt_state, params, _ = optimizer.step(params)
            optimizer = optimizer.post_step(opt_state, params)
        return params


@chex.dataclass(frozen=True)
class MultiOptimizer(Optimizer, ABC):
    """Optimization of a list of objectives using a list of simulators.

    This abstract class implements the logic to coordinate multiple objectives
    and multiple simulators asynchronously, using the interface to remote
    simulators and objectives defined in this library. `*_async` methods of
    those should return some type of future object - concrete implementations of
    this class must implement the methods to wait for and get results from those
    futures.

    Parameters:
        objectives: A list of objectives to optimize.
        simulators: A list of simulators to use for the optimization.
        aggregate_grad_fn: A function that aggregates the gradients from the objectives.
        optimizer: An optax optimizer.
        optimizer_state: The state of the optimizer.
        logger: A logger to use for the optimization.
    """

    objectives: list[jdna_objective.AsyncObjective]
    simulators: list[AsyncSimulation]
    aggregate_grad_fn: typing.Callable[[list[jdna_types.Grads]], jdna_types.Grads]
    optimizer: optax.GradientTransformation
    optimizer_state: optax.OptState | None = None
    logger: jdna_logger.Logger = dc.field(default_factory=jdna_logger.NullLogger)
    _sim_exposes: dict[typing.Any, list[str]] = dc.field(init=False)
    _expose_map: dict[str, typing.Any] = dc.field(init=False)

    def __post_init__(self) -> None:
        """Validate the initialization of the Optimization."""
        if not all(hasattr(obj, "calculate_async") for obj in self.objectives):
            raise ValueError("All objectives must implement calculate_async.")
        if not all(hasattr(obj, "run_async") for obj in self.simulators):
            raise ValueError("All simulators must implement run_async.")

        # cache the provided observables for each simulator, since this
        # information may make a remote call and is not expected to change.
        # We need a lookup in both directions, to find what sims provide what,
        # but also to simply map from observable to simulator.
        object.__setattr__(self, "_sim_exposes", {sim: sim.exposes() for sim in self.simulators})
        object.__setattr__(self, "_expose_map", {exp: sim for sim, exps in self._sim_exposes.items() for exp in exps})

        if len(self._expose_map) != sum(len(exps) for exps in self._sim_exposes.values()):
            raise ValueError("Multiple simulators expose the same observable.")

    @abstractmethod
    def futures_wait(self, futures: list[typing.Any]) -> tuple[list[typing.Any], list[typing.Any]]:
        """Waits for the given futures to complete.

        Args:
            futures: A list of futures to wait for.
        """

    @abstractmethod
    def future_get(self, future: typing.Any) -> typing.Any:
        """Gets the result of the given future locally.

        Args:
            future: The future to get the result from.
        """

    @abstractmethod
    def future_pass(self, future: typing.Any) -> typing.Any:
        """Get the future for passing to other async calls.

        Args:
            future: The future to pass.
        """

    def step(self, params: jdna_types.Params) -> OptResult:  # noqa: C901
        """Perform a single optimization step.

        Args:
            params: The current parameters.

        Returns:
            A tuple containing the updated optimizer state, new params, and the gradients.
        """
        # maps futures to producer (simulator observation or objective). This is
        # useful so we know where to route results when they complete without
        # having to encode this information in the future itself.
        remote_map = {}
        # maps objective to its needed observables to avoid repetitive calls
        needs_map = {}
        # maps objective to gradients that have been calculated, used to keep order and done-ness
        grads_map = {}

        def schedule_simulator(sim: BaseSimulation) -> None:
            exposures = self._sim_exposes[sim]
            if set(exposures).isdisjoint(remote_map.values()):
                refs = sim.run_async(params)
                refs = [refs] if len(exposures) == 1 else refs
                for ref, exp in zip(refs, exposures, strict=True):
                    remote_map[ref] = exp

        def route_to_objective(obs_name: str, ref: typing.Any) -> None:
            for objective, needs in needs_map.items():
                if obs_name in needs:
                    objective.update_one(obs_name, ref)
                    needs_map[objective].remove(obs_name)

        while needed_objectives := set(self.objectives) - set(grads_map):
            for objective in needed_objectives:
                # If we have already scheduled and are still waiting for
                # observables we know it needs, then no need to check again.
                if needs_map.get(objective):
                    continue
                # Not every objective needs updates for each step, for example
                # DiffTRe can reuse observables in the right conditions.
                if objective.is_ready():
                    remote_map[objective.calculate_async()] = objective
                else:
                    needs_map[objective] = objective.needed_observables()
                    # based on needed observables, find which sims provide from
                    # map, then schedule those sims (if not already scheduled)
                    for sim in {self._expose_map[obs] for obs in needs_map[objective]}:
                        schedule_simulator(sim)

            done, _ = self.futures_wait(list(remote_map.keys()))

            for ref in done:
                producer = remote_map.pop(ref)
                if producer in self.objectives:
                    grads_map[producer] = self.future_get(ref)
                else:
                    route_to_objective(producer, self.future_pass(ref))

        grads = self.aggregate_grad_fn(list(grads_map.values()))
        return self.get_updates_and_state(grads, params)

    def post_step(
        self,
        optimizer_state: optax.OptState,
        opt_params: jdna_types.Params,
    ) -> "Optimizer":
        """An update step intended to be called after an optimization step."""
        for objective in self.objectives:
            objective.post_step(opt_params)
        return self.replace(optimizer_state=optimizer_state)


class RayMultiOptimizer(MultiOptimizer):
    """An optimization that uses Ray actors for objectives and simulators."""

    @override
    def futures_wait(self, futures: list[typing.Any]) -> tuple[list[typing.Any], list[typing.Any]]:
        return ray.wait(futures, fetch_local=False, num_returns=1)

    @override
    def future_get(self, future: typing.Any) -> typing.Any:
        return ray.get(future)

    @override
    def future_pass(self, future: typing.Any) -> typing.Any:
        return future


@chex.dataclass(frozen=True)
class SimpleOptimizer(Optimizer):
    """A simple optimizer that uses a single objective and simulator."""

    objective: jdna_objective.Objective
    simulator: BaseSimulation
    optimizer: optax.GradientTransformation
    optimizer_state: optax.OptState | None = None
    logger: jdna_logger.Logger = dc.field(default_factory=jdna_logger.NullLogger)

    def step(self, params: jdna_types.Params) -> tuple[optax.OptState, list[jdna_types.Grads], list[jdna_types.Grads]]:
        """Perform a single optimization step.

        Args:
            params: The current parameters.

        Returns:
            A tuple containing the updated optimizer state, new params, and the gradients.
        """
        # get the currently needed observables
        # some objectives might use difftre and not actually need something rerun
        # so check which objectives have observables that need to be run
        if self.objective.is_ready():
            grads = self.objective.calculate()
        else:
            observables = self.simulator.run(params)
            exposes = self.simulator.exposes()
            observables = [observables] if len(exposes) == 1 else observables
            self.objective.update(exposes, *observables)
            grads = self.objective.calculate()

        return self.get_updates_and_state(grads, params)

    def post_step(self, optimizer_state: optax.OptState, opt_params: jdna_types.Params) -> "SimpleOptimizer":
        """An update step intended to be called after an optimization step."""
        self.objective.post_step(opt_params)
        return self.replace(optimizer_state=optimizer_state)
