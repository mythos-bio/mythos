"""Objectives implemented as ray actors."""

import functools
import logging
import math
import operator
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable
import uuid

import jax
import jax.numpy as jnp
import jax_md
import ray
from typing_extensions import override

import mythos.energy as jdna_energy
import mythos.utils.types as jdna_types
from mythos.energy.base import EnergyFunction
from mythos.simulators.io import SimulatorTrajectory

ERR_DIFFTRE_MISSING_KWARGS = "Missing required kwargs: {missing_kwargs}."
ERR_MISSING_ARG = "Missing required argument: {missing_arg}."
ERR_OBJECTIVE_NOT_READY = "Not all required observables have been obtained."

EnergyFn = jdna_energy.base.BaseEnergyFunction | jdna_energy.base.ComposedEnergyFunction
empty_dict = types.MappingProxyType({})


class Objective:
    """Base class for objectives that calculate gradients."""

    def __init__(
        self,
        name: str,
        required_observables: list[str],
        logging_observables: list[str],
        grad_or_loss_fn: typing.Callable[[tuple[str, ...]], tuple[jdna_types.Grads, list[tuple[str, typing.Any]]]],
        logger_config: dict[str, typing.Any] = empty_dict,
    ) -> "Objective":
        """Initialize the objective.

        Args:
            name (str): The name of the objective.
            required_observables (list[str]): The observables that are required
                to calculate the gradients.
            logging_observables (list[str]): The observables that are used for
                logging.
            grad_or_loss_fn (typing.Callable[[tuple[str, ...]], jdna_types.Grads]):
                The function that calculates the loss of the objective
            logger_config (dict[str, typing.Any]): The configuration for the logger.
        """
        if name is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="name"))
        if required_observables is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="required_observables"))
        if logging_observables is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="logging_observables"))
        if grad_or_loss_fn is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="grad_or_loss_fn"))

        self._name = name
        self._required_observables = required_observables
        self._grad_or_loss_fn = grad_or_loss_fn
        self._obtained_observables = {}
        self._logging_observables = logging_observables
        logging.basicConfig(**logger_config)
        self._logger = logging.getLogger(__name__)

    def needed_observables(self) -> set[str]:
        """Return the observables that are still needed."""
        return set(self._required_observables) - set(self._obtained_observables)

    def logging_observables(self) -> dict[str, typing.Any]:
        """Return the latest observed values for the logging observables."""
        return {k: v for k, v in self._obtained_observables.items() if k in self._logging_observables}

    def is_ready(self) -> bool:
        """Check if the objective is ready to calculate its gradients."""
        return len(self.needed_observables()) == 0

    def update_one(self, name: str, value: typing.Any) -> None:
        """Update a single observable with the latest simulation result."""
        if name in self._required_observables:
            self._obtained_observables[name] = value

    def update(self, names: list[str], *values: typing.Any) -> None:
        """Update several observables with the latest simulation results."""
        for name, value in zip(names, values, strict=True):
            self.update_one(name, value)

    def _get_observables_sorted(self) -> list[typing.Any]:
        return [self._obtained_observables[name] for name in self._required_observables]

    def calculate(self) -> list[jdna_types.Grads]:
        """Calculate the gradients of the objective."""
        if not self.is_ready():
            raise ValueError(ERR_OBJECTIVE_NOT_READY)

        grads, aux = self._grad_or_loss_fn(*self._get_observables_sorted())
        self._obtained_observables.update(dict(aux))
        return grads

    def post_step(self, opt_params: dict) -> None:  # noqa: ARG002 - not all objectives need params
        """Reset the needed observables for the next step."""
        self._obtained_observables.clear()


def compute_weights_and_neff(
    beta: float,
    new_energies: jdna_types.Arr_N,
    ref_energies: jdna_types.Arr_N,
) -> tuple[jnp.ndarray, float]:
    """Compute the weights and normalized effective sample size of a trajectory.

    Calculation derived from the DiffTRe algorithm.

    https://www.nature.com/articles/s41467-021-27241-4
    See equations 4 and 5.

    Args:
        beta: The inverse temperature.
        new_energies: The new energies of the trajectory.
        ref_energies: The reference energies of the trajectory.

    Returns:
        The weights and the normalized effective sample size
    """
    diffs = new_energies - ref_energies
    boltz = jnp.exp(-beta * diffs)
    weights = boltz / jnp.sum(boltz)
    n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
    return weights, n_eff / len(weights)


def compute_loss(
    opt_params: jdna_types.Params,
    energy_fn: EnergyFunction,
    beta: float,
    loss_fn: Callable[
        [jax_md.rigid_body.RigidBody, jdna_types.Arr_N, EnergyFn], tuple[jnp.ndarray, tuple[str, typing.Any]]
    ],
    ref_states: jax_md.rigid_body.RigidBody,
    ref_energies: jdna_types.Arr_N,
    observables: list[typing.Any],
) -> tuple[float, tuple[float, jnp.ndarray]]:
    """Compute the grads, loss, and auxiliary values.

    Args:
        opt_params: The optimization parameters.
        energy_fn: The energy function.
        beta: The inverse temperature.
        loss_fn: The loss function.
        ref_states: The reference states of the trajectory.
        ref_energies: The reference energies of the trajectory.
        observables: The observables passed to the loss function.

    Returns:
        The grads, the loss, a tuple containing the normalized effective sample
        size and the measured value of the trajectory, and the new energies.
    """
    energy_fn = energy_fn.with_params(opt_params)
    new_energies = energy_fn.map(ref_states.rigid_body)
    weights, neff = compute_weights_and_neff(
        beta,
        new_energies,
        ref_energies,
    )
    loss, (measured_value, meta) = loss_fn(ref_states, weights, energy_fn, opt_params, observables)
    return loss, (neff, measured_value, new_energies)


compute_loss_and_grad = jax.value_and_grad(compute_loss, has_aux=True)


class DiffTReObjective(Objective):
    """Objective that calculates the gradients of an objective using DiffTRe."""

    def __init__(
        self,
        name: str,
        required_observables: list[str],
        logging_observables: list[str],
        grad_or_loss_fn: typing.Callable[[tuple[jdna_types.SimulatorActorOutput]], jdna_types.Grads],
        energy_fn: EnergyFunction,
        opt_params: jdna_types.Params,
        beta: float,
        n_equilibration_steps: int,
        min_n_eff_factor: float = 0.95,
        max_valid_opt_steps: int = math.inf,
        logging_config: dict[str, typing.Any] = empty_dict,
    ) -> "DiffTReObjective":
        """Initialize the DiffTRe objective.

        Args:
            name: The name of the objective.
            required_observables: The observables that are required to calculate the gradients.
            logging_observables: The observables that are used for logging.
            grad_or_loss_fn: The function that calculates the loss of the objective.
            energy_fn: The energy function used to compute energies.
            opt_params: The optimization parameters.
            beta: The inverse temperature.
            n_equilibration_steps: The number of equilibration steps.
            min_n_eff_factor: The minimum normalized effective sample size.
            max_valid_opt_steps: The maximum number of steps a trajectory is valid for.
            logging_config: The configuration for the logger.
        """
        super().__init__(
            name,
            required_observables,
            logging_observables,
            grad_or_loss_fn,
            logger_config=logging_config,
        )
        if energy_fn is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="energy_fn"))
        if opt_params is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="opt_params"))
        if beta is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="beta"))
        if n_equilibration_steps is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="n_equilibration_steps"))

        self._energy_fn = energy_fn
        self._opt_params = opt_params
        self._beta = beta
        self._n_eq_steps = n_equilibration_steps
        self._n_eff_factor = min_n_eff_factor
        self._max_valid_opt_steps = max_valid_opt_steps
        self._opt_steps = 1

        self._reference_states = None
        self._reference_energies = None

    @override
    def calculate(self) -> list[jdna_types.Grads]:
        if not self.is_ready():
            raise ValueError(ERR_OBJECTIVE_NOT_READY)

        sorted_obs = self._get_observables_sorted()

        (loss, (_, measured_value, new_energies)), grads = compute_loss_and_grad(
            self._opt_params,
            self._energy_fn,
            self._beta,
            self._grad_or_loss_fn,
            self._reference_states,
            self._reference_energies,
            sorted_obs,
        )

        self._obtained_observables.update({"loss": loss})
        self._obtained_observables.update(dict([measured_value]))
        return grads

    @override
    def is_ready(self) -> bool:
        have_trajectories = super().is_ready()
        if have_trajectories:
            sorted_obs = self._get_observables_sorted()
            trajectories = [obs for obs in sorted_obs if isinstance(obs, SimulatorTrajectory)]

            if self._reference_states is None:
                def slc_f(n: int) -> slice:
                    return slice(self._n_eq_steps, n, None)

                self._reference_states = functools.reduce(
                    operator.add,
                    [obs.slice(slc_f(len(obs.rigid_body.center))) for obs in trajectories],
                )

                self._reference_energies = self._energy_fn.with_params(self._opt_params).map(
                    self._reference_states.rigid_body
                )

            self._logger.info("trajectory length is %d", len(self._reference_states.rigid_body.center))

            _, neff = compute_weights_and_neff(
                beta=self._beta,
                new_energies=self._energy_fn.with_params(self._opt_params).map(self._reference_states.rigid_body),
                ref_energies=self._reference_energies,
            )

            self._obtained_observables["neff"] = neff

            # if the trajectory is no longer valid remove it form obtained
            # and add it to needed so that a new trajectory is run.
            self._logger.info("checking neff %f neff_factory %f", neff, self._n_eff_factor)
            self._logger.info("checking opt steps %d vs %f", self._opt_steps, float(self._max_valid_opt_steps))
            if (neff < self._n_eff_factor) or (self._opt_steps == self._max_valid_opt_steps):
                self._obtained_observables.clear()
                self._reference_states = None
                self._opt_steps = 1
                have_trajectories = False

        return have_trajectories

    @override
    def post_step(
        self,
        opt_params: jdna_types.Params,
    ) -> None:
        # DiffTre objectives may not need to update the trajectory depending on neff
        # the need for a new trajectory is checked in `is_ready`
        self._obtained_observables.pop("neff", None)
        self._obtained_observables.pop("loss", None)
        self._opt_params = opt_params
        self._opt_steps += 1


class AsyncObjective(Objective, ABC):
    """An abstract base class for asynchronous objectives."""

    @abstractmethod
    def calculate_async(self) -> typing.Any:
        """Calculate the gradients asynchronously."""


class RayObjective(Objective):
    """Objective that manages and dispatches to a remote objective in Ray."""

    def __init__(self, objective_class: type[Objective], *args, **kwds) -> "RayObjective":
        """Initialize the Ray objective."""
        self._name = f"Ray({objective_class}).{uuid.uuid4()}" if "name" not in kwds else kwds["name"]
        self.objective = ray.remote(objective_class).remote(*args, **kwds)

    @override
    def update(self, *args, **kwargs) -> None:
        ray.get(self.objective.update.remote(*args, **kwargs))

    @override
    def update_one(self, name: str, value: typing.Any) -> None:
        ray.get(self.objective.update_one.remote(name, value))

    @override
    def needed_observables(self) -> set[str]:
        return ray.get(self.objective.needed_observables.remote())

    @override
    def is_ready(self) -> bool:
        return ray.get(self.objective.is_ready.remote())

    @override
    def calculate(self) -> list[jdna_types.Grads]:
        return ray.get(self.objective.calculate.remote())

    @override
    def post_step(self, opt_params: dict) -> None:
        ray.get(self.objective.post_step.remote(opt_params))

    @override
    def logging_observables(self) -> dict[str, typing.Any]:
        return ray.get(self.objective.logging_observables.remote())

    @override
    def calculate_async(self) -> ray.ObjectRef:
        return self.objective.calculate.remote()

