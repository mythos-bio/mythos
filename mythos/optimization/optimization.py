"""Runs an optimization loop using Ray actors for objectives and simulators."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import field
from typing import Any

import chex
import optax
import ray
from ray import ObjectRef as RayRef
from typing_extensions import override

from mythos.optimization.objective import Objective, ObjectiveOutput
from mythos.simulators.base import Simulator
from mythos.ui.loggers import logger as jdna_logger
from mythos.utils.scheduler import SchedulerUnit
from mythos.utils.types import Grads, Params

ERR_MISSING_OBJECTIVES = "At least one objective is required."
ERR_MISSING_SIMULATORS = "At least one simulator is required."
ERR_MISSING_AGG_GRAD_FN = "An aggregate gradient function is required."
ERR_MISSING_OPTIMIZER = "An optimizer is required."
# to prevent infinite unresolvable loops in step. The first call may use cached
# observables, so may required rerun of sims. After this, we don't expect any
# new information, so not-ready state after this is an error.
OBJECTIVE_PER_STEP_CALL_LIMIT = 2


@chex.dataclass(frozen=True, kw_only=True)
class OptimizerState:
    """State container for optimization loops.

    This dataclass stores all mutable state needed during optimization,
    allowing optimizers to work with frozen objective dataclasses.

    Attributes:
        observables: Current observable values from simulators.
        state: Per-objective/simulator state (keyed by name).
            Both objective and simulator state share this namespace,
            so names should be unique across objectives and simulators.
        optimizer_state: Current optax optimizer state.
    """

    observables: dict[str, Any] = field(default_factory=dict)
    component_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    optimizer_state: Any | None = None  # optax.OptState


@chex.dataclass(frozen=True, kw_only=True)
class OptimizerOutput:
    """Output container for optimization steps.

    Attributes:
        grads: The computed (aggregate) gradients from the optimization step.
        opt_params: The updated parameters after the optimization step.
        state: The updated optimizer state after the optimization step. This
            data structure should be passed back into the next call to step.
        observables: The logged observables from the optimization step. These
            are keyed by component name (e.g. objective) and each value should
            itself be a dict of observable name to value.
    """
    grads: Grads
    opt_params: Params
    state: OptimizerState
    observables: dict[str, dict[str, Any]] = field(default_factory=dict)


@chex.dataclass(frozen=True, kw_only=True)
class Optimizer(ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def step(self, params: Params, state: OptimizerState | None = None) -> OptimizerOutput:
        """Perform a single optimization step.

        Args:
            params: The current parameters.
            state: The current optimization state. If None, an empty state is initialized.

        Returns:
            An optimizer output including params, new state, grads, and observables.
        """


@chex.dataclass(frozen=True, kw_only=True)
class RayOptimizer(Optimizer):
    """Optimization of a list of objectives using a list of simulators.

    Parameters:
        objectives: A list of objectives to optimize.
        simulators: A list of simulators to use for the optimization.
        aggregate_grad_fn: A function that aggregates the gradients from the objectives.
        optimizer: An optax optimizer.
        optimizer_state: The state of the optimizer.
        logger: A logger to use for the optimization.
        remote_options_default: Default Ray options to apply to all remote calls.
    """

    objectives: list[Objective]
    simulators: list[Simulator]
    aggregate_grad_fn: Callable[[list[Grads]], Grads]
    optimizer: optax.GradientTransformation
    logger: jdna_logger.Logger = field(default_factory=jdna_logger.NullLogger)
    remote_options_default: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the initialization of the Optimization."""
        if not self.objectives:
            raise ValueError(ERR_MISSING_OBJECTIVES)

        if not self.simulators:
            raise ValueError(ERR_MISSING_SIMULATORS)

        if self.aggregate_grad_fn is None:
            raise ValueError(ERR_MISSING_AGG_GRAD_FN)

        if self.optimizer is None:
            raise ValueError(ERR_MISSING_OPTIMIZER)

        # Check for conflicts in global namespaces that we use for coordination
        all_names = [obj.name for obj in self.objectives] \
            + [sim.name for sim in self.simulators] \
            + [exp for sim in self.simulators for exp in sim.exposes()]
        if len(all_names) != len(set(all_names)):
            raise ValueError("All objective, simulator, and exposes names must be unique")

    def _create_and_run_remote(self, fun: callable, ray_options: dict, *args) -> RayRef|list[RayRef]:
        remote_fun = ray.remote(fun).options(**ray_options)
        return remote_fun.remote(*args)

    def _get_ray_options(self, unit: SchedulerUnit) -> dict[str, Any]:
        options = {}
        if unit_hints := getattr(unit, "scheduler_hints", None):
            options = unit_hints.to_dict(engine="ray", rewrite_options={"mem_mb": "memory"})
            if "memory" in options:
                options["memory"] = int(options["memory"] * 1024 * 1024)  # Ray expects bytes
        return {**self.remote_options_default, **options}

    def _run_simulator(
            self, simulator: Simulator, params: Params, **state
        ) -> tuple[list[RayRef], RayRef]:
        def simulator_run_fn(params: Params, state: dict[str, Any]) -> tuple[list[RayRef], RayRef]:
            output = simulator.run(opt_params=params, **state)
            return *output.observables, output.state

        ray_opts = {
            **self._get_ray_options(simulator),
            "name": "simulator_run:" + simulator.name,
            "num_returns": 1 + len(simulator.exposes()),
        }
        refs = self._create_and_run_remote(simulator_run_fn, ray_opts, params, state)
        return refs[:-1], refs[-1]  # observables as a list, state

    def _run_objective(
            self, objective: Objective, observables: dict[str, RayRef], params: Params, **state
        ) -> RayRef:
        def objective_compute_fn(obs: dict[str, RayRef], params: Params, state: dict[str, Any]) -> ObjectiveOutput:
            obs = {k: ray.get(v) for k, v in obs.items()}
            return objective.calculate(observables=obs, opt_params=params, **state)

        ray_opts = {
            **self._get_ray_options(objective),
            "name": "objective_compute:" + objective.name,
        }
        return self._create_and_run_remote(objective_compute_fn, ray_opts, observables, params, state)

    def _wait_remotes(self, refs: list[RayRef]) -> list[RayRef]:
            ref_list = list(refs)
            ray.wait(ref_list, fetch_local=False, num_returns=1)
            # The below is to maximize our chance of getting multiple at once
            # (for example multiple observables and state from a simulator)
            ready, _ = ray.wait(ref_list, fetch_local=False, timeout=0.1)
            return ready

    @override
    def step(self, params: Params, state: OptimizerState|None = None) -> OptimizerOutput:  # noqa: C901, PLR0912
        state = state or OptimizerState()
        state_observables, component_state = state.observables.copy(), state.component_state.copy()

        obj_lookup = {obj.name: obj for obj in self.objectives}
        call_count = {obj.name: 0 for obj in self.objectives}
        sim_lookup = {sim.name: sim for sim in self.simulators}
        expose_lookup = {exp: sim for sim in self.simulators for exp in sim.exposes()}
        ref_map, grads_completed, output_observables = {}, {}, {}

        # schedule all objectives that already have their observables in state
        while needed_objectives := set(obj_lookup) - set(grads_completed):
            for obj_name in needed_objectives:
                objective = obj_lookup[obj_name]
                # skip if we are currently running it
                if objective.name in ref_map.values():
                    continue
                # It is an unresolvable state if we have called the objective
                # more than twice
                if call_count[objective.name] > OBJECTIVE_PER_STEP_CALL_LIMIT:
                    raise RuntimeError(f"Objective {objective.name} could not be resolved after multiple attempts.")
                # If we have all the observables in state, we make an attempt at
                # the objective. This may return a not ready signal, in which
                # case observables will be cleared to trigger this logic again.
                if set(objective.required_observables).issubset(state_observables):
                    obj_observables = {k: state_observables[k] for k in objective.required_observables}
                    obj_state = component_state.get(objective.name, {})
                    ref = self._run_objective(objective, obj_observables, params, **obj_state)
                    ref_map[ref] = objective.name
                    call_count[objective.name] += 1
                # there are simulators running that provide some of what we
                # need, so we have gone through the scheduling step
                elif set(objective.required_observables).intersection(ref_map.values()):
                    continue
                else:
                    needed_sims = {expose_lookup[exp].name for exp in objective.required_observables}
                    # filter out sims we know are running based on state ref
                    for sim_name in needed_sims - set(ref_map.values()):
                        sim = sim_lookup[sim_name]
                        # make sure we aren't waiting on any of the observables
                        # this provides. It may be possible that the ref of
                        # state or some observables become available separately
                        if set(sim.exposes()).intersection(ref_map.values()):
                            continue
                        sim_state = component_state.get(sim.name, {})
                        refs, md_ref = self._run_simulator(sim, params, **sim_state)
                        for r, exp in zip(refs, sim.exposes(), strict=True):
                            ref_map[r] = exp
                        ref_map[md_ref] = sim.name

            # wait for anything to finish. We do a second wait without num
            # returns but non-blocking to gather as many as we can at once
            ready = self._wait_remotes(ref_map.keys())

            for ref in ready:
                producer = ref_map.pop(ref)
                if producer in obj_lookup:
                    output = ray.get(ref)
                    component_state[producer] = output.state
                    if output.is_ready:
                        grads_completed[producer] = output.grads
                        output_observables[producer] = output.observables
                    else:
                        # remove the needs from the state observables so the
                        # above loop check will schedule the providing simulator
                        state_observables = {k: v for k, v in state.observables.items() if k not in output.needs_update}
                elif producer in expose_lookup:
                    state_observables[producer] = ref
                else:  # finally it must be simulator state
                    component_state[producer] = ray.get(ref)

        grads = self.aggregate_grad_fn(list(grads_completed.values()))
        opt_state = state.optimizer_state or self.optimizer.init(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return OptimizerOutput(
            opt_params=new_params,
            state=state.replace(
                optimizer_state=opt_state,
                component_state=component_state,
                observables=state_observables,
            ),
            grads=grads,
            observables=output_observables,
        )


@chex.dataclass(frozen=True)
class SimpleOptimizer(Optimizer):
    """A simple optimizer that uses a single objective and simulator.

    This optimizer manages the state for a frozen Objective dataclass,
    passing observables and state through the compute method.
    State is managed via OptimizationState which is passed in and out.
    """

    objective: Objective
    simulator: Simulator
    optimizer: optax.GradientTransformation
    logger: jdna_logger.Logger = field(default_factory=jdna_logger.NullLogger)

    @override
    def step(self, params: Params, state: OptimizerState | None = None) -> OptimizerOutput:
        state = state or OptimizerState()
        obj_state = state.component_state.get(self.objective.name, {})
        sim_state = state.component_state.get(self.simulator.name, {})
        obj_output = None

        if state.observables:
            obj_output = self.objective.calculate(state.observables, opt_params=params, **obj_state)
            obj_state = obj_output.state

        if obj_output is None or not obj_output.is_ready:
            sim_output = self.simulator.run(params, **sim_state)
            sim_state = sim_output.state
            exposes = self.simulator.exposes()
            state = state.replace(observables=dict(zip(exposes, sim_output.observables, strict=True)))

            # Try again with updated observables
            obj_output = self.objective.calculate(state.observables, opt_params=params, **obj_state)
            obj_state = obj_output.state

            if not obj_output.is_ready:
                # this should be an impossible state, could end up in infinite loop
                raise ValueError("Objective readiness check failed after simulation run.")

        grads = obj_output.grads
        opt_state = state.optimizer_state or self.optimizer.init(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # should this be filtered? also be allowed to return filtered from
        # simulators?
        output_observables = {self.objective.name: obj_output.observables}

        return OptimizerOutput(
            opt_params=new_params,
            state=state.replace(
                optimizer_state=opt_state,
                component_state={
                    **state.component_state,
                        self.objective.name: obj_state,
                        self.simulator.name: sim_state,
                    },
            ),
            grads=grads,
            observables=output_observables
        )

