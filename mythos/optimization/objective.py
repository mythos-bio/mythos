"""Objectives implemented as frozen chex dataclasses.

Objectives are immutable dataclasses that compute gradients from observables.
State is passed through the compute method and returned in the ObjectiveOutput.
"""

import math
import types
import typing
from collections.abc import Callable
from dataclasses import field

import chex
import jax
import jax.numpy as jnp
import jax_md

import mythos.energy as jdna_energy
import mythos.utils.types as jdna_types
from mythos.energy.base import EnergyFunction
from mythos.simulators.io import SimulatorTrajectory
from mythos.utils.scheduler import SchedulerUnit

ERR_DIFFTRE_MISSING_KWARGS = "Missing required kwargs: {missing_kwargs}."
ERR_MISSING_ARG = "Missing required argument: {missing_arg}."
ERR_OBJECTIVE_NOT_READY = "Not all required observables have been obtained."

EnergyFn = jdna_energy.base.BaseEnergyFunction | jdna_energy.base.ComposedEnergyFunction
empty_dict = types.MappingProxyType({})


@chex.dataclass(frozen=True, kw_only=True)
class ObjectiveOutput:
    """Output of an objective calculation.

    Attributes:
        is_ready: Whether the objective has computed gradients.
        grads: The computed gradients, if ready.
        observables: Observable values to preserve between calls.
        state: State information to pass back to the next compute call.
            For DiffTRe, this includes reference_states, reference_energies, opt_steps.
        needs_update: List of observable names that need new values.
    """

    is_ready: bool
    grads: jdna_types.Grads | None = None
    observables: dict[str, typing.Any] = field(default_factory=dict)
    state: dict[str, typing.Any] = field(default_factory=dict)
    needs_update: tuple[str, ...] = field(default_factory=tuple)


@chex.dataclass(frozen=True, kw_only=True)
class Objective(SchedulerUnit):
    """Frozen dataclass for objectives that calculate gradients.

    Objectives are immutable - all state is passed in and out through the
    calculate method. The ObjectiveOutput.state field carries state that
    needs to persist between calculate calls (e.g., reference states for DiffTRe).

    Attributes:
        name: The name of the objective.
        required_observables: Observable names required to compute gradients.
        logging_observables: Observable names used for logging.
        grad_or_loss_fn: Function that computes gradients from observables.
        scheduler_hints: Optional hints for scheduling this objective.
    """

    name: str
    required_observables: tuple[str, ...]
    logging_observables: tuple[str, ...] = field(default_factory=tuple)
    grad_or_loss_fn: typing.Callable[
        [tuple[typing.Any, ...]], tuple[jdna_types.Grads, list[tuple[str, typing.Any]]]
    ] = field(repr=False)

    def __post_init__(self) -> None:
        """Validate required fields."""
        if self.name is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="name"))
        if self.required_observables is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="required_observables"))
        if self.grad_or_loss_fn is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="grad_or_loss_fn"))

    def calculate(
        self,
        observables: dict[str, typing.Any],
        opt_params: jdna_types.Params | None = None,  # noqa: ARG002 - base class ignores opt_params
        **_kwargs,
    ) -> ObjectiveOutput:
        """Compute gradients from observables.

        Args:
            observables: Dictionary mapping observable names to their values.
            opt_params: Current optimization parameters (unused in base class).

        Returns:
            ObjectiveOutput containing gradients and updated state.
        """
        # Check if all required observables are present
        missing = [obs for obs in self.required_observables if obs not in observables]
        if missing:
            return ObjectiveOutput(
                is_ready=False,
                needs_update=tuple(missing),
            )

        # Sort observables in the required order
        sorted_obs = [observables[key] for key in self.required_observables]

        grads, aux = self.grad_or_loss_fn(*sorted_obs)

        # Build output observables from aux and input observables
        output_observables = dict(aux)
        output_observables.update(dict(zip(self.required_observables, sorted_obs, strict=True)))

        return ObjectiveOutput(
            is_ready=True,
            grads=grads,
            observables=output_observables,
            state={},
            needs_update=(),
        )

    def get_logging_observables(
        self,
        observables: dict[str, typing.Any],
    ) -> list[tuple[str, typing.Any]]:
        """Return the observable values for logging.

        Args:
            observables: Dictionary mapping observable names to their values.

        Returns:
            List of (name, value) tuples for logging observables.
        """
        return [(name, observables[name]) for name in self.logging_observables if name in observables]


def compute_weights_and_neff(
    beta: jdna_types.Arr_N | float,
    new_energies: jdna_types.Arr_N,
    ref_energies: jdna_types.Arr_N,
) -> tuple[jnp.ndarray, float]:
    """Compute the weights and normalized effective sample size of a trajectory.

    Calculation derived from the DiffTRe algorithm.

    https://www.nature.com/articles/s41467-021-27241-4
    See equations 4 and 5.

    Args:
        beta: The inverse temperature. May be a scalar or a per-state array.
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


def compute_min_segment_neff(
    temperature: jnp.ndarray,
    new_energies: jdna_types.Arr_N,
    ref_energies: jdna_types.Arr_N,
) -> float:
    """Compute the minimum normalized effective sample size across temperature segments.

    For trajectories that span multiple temperatures, each temperature segment
    has its own reweighting statistics.  This function computes the per-segment
    ``n_eff`` and returns the minimum, which is the binding constraint for
    deciding whether the trajectory is still valid for reweighting.

    Args:
        temperature: Per-state temperature array in kT (simulation energy
            units), shape ``(n_states,)``.
        new_energies: Energies under the current parameters, shape
            ``(n_states,)``.
        ref_energies: Energies under the reference parameters, shape
            ``(n_states,)``.

    Returns:
        The minimum ``n_eff`` across all temperature segments.
    """

    def segment_neff(temp: float) -> float:
        mask = temperature == temp
        _, neff = compute_weights_and_neff(1.0 / temp, new_energies[mask], ref_energies[mask])
        return float(neff)

    return min(segment_neff(t) for t in jnp.unique(temperature))


def compute_loss(
    opt_params: jdna_types.Params,
    energy_fn: EnergyFunction,
    beta: jdna_types.Arr_N | float,
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
        beta: The inverse temperature. May be a scalar or a per-state array.
        loss_fn: The loss function.
        ref_states: The reference states of the trajectory.
        ref_energies: The reference energies of the trajectory.
        observables: The observables passed to the loss function.

    Returns:
        The grads, the loss, a tuple containing the normalized effective sample
        size and the measured value of the trajectory, and the new energies.
    """
    energy_fn = energy_fn.with_params(opt_params)
    new_energies = energy_fn.map(ref_states)
    weights, neff = compute_weights_and_neff(
        beta,
        new_energies,
        ref_energies,
    )
    loss, (measured_value, meta) = loss_fn(ref_states, weights, energy_fn, opt_params, observables)
    return loss, (neff, measured_value, new_energies)


compute_loss_and_grad = jax.value_and_grad(compute_loss, has_aux=True)


@chex.dataclass(frozen=True, kw_only=True)
class DiffTReObjective(Objective):
    """Frozen dataclass for DiffTRe-based gradient computation.

    DiffTRe (Differentiable Trajectory Reweighting) allows computing gradients
    by reweighting trajectories rather than running new simulations. State such
    as reference_states, reference_energies, and opt_steps is passed through
    the metadata field of ObjectiveOutput.

    Temperature is obtained from the ``SimulatorTrajectory.temperature`` field
    (per-state kT in simulation energy units).  The inverse temperature
    ``beta = 1 / temperature`` is used for Boltzmann reweighting.  When
    multiple temperature segments are present the minimum per-segment
    ``n_eff`` is used as the validity criterion.

    Attributes:
        energy_fn: The energy function used to compute energies.
        n_equilibration_steps: Number of equilibration steps (snapshot states, not timesteps) to skip.
        min_n_eff_factor: Minimum normalized effective sample size threshold.
        max_valid_opt_steps: Maximum optimization steps before requiring new trajectory.
    """

    energy_fn: EnergyFunction = field(repr=False)
    n_equilibration_steps: int = 0
    min_n_eff_factor: float = 0.95
    max_valid_opt_steps: float = math.inf

    def __post_init__(self) -> None:
        """Validate required fields."""
        Objective.__post_init__(self)
        if self.energy_fn is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="energy_fn"))
        if self.n_equilibration_steps is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="n_equilibration_steps"))
        if self.n_equilibration_steps < 0:
            raise ValueError(f"n_equilibration_steps must be non-negative, got {self.n_equilibration_steps}.")
        if self.max_valid_opt_steps <= 0:
            raise ValueError("max_valid_opt_steps must be positive or infinity.")

    def calculate(
        self,
        observables: dict[str, typing.Any],
        opt_params: jdna_types.Params,
        opt_steps: int = 0,
        reference_opt_params: jdna_types.Params | None = None,
    ) -> ObjectiveOutput:
        """Compute gradients using DiffTRe reweighting.

        Args:
            observables: Dictionary mapping observable names to their values.
            opt_params: Current optimization parameters for energy computation.
            opt_steps: Current optimization step count.
            reference_opt_params: Optimization parameters used to compute
                reference energies.

        Returns:
            ObjectiveOutput with gradients and updated metadata.
        """
        # Short-circuit: if we've exceeded max optimization steps, request
        # a new trajectory immediately without any computation.
        if opt_steps >= self.max_valid_opt_steps:
            return ObjectiveOutput(
                is_ready=False,
                needs_update=tuple(self.required_observables),
                state={"opt_steps": 0},
            )

        # Check if all required observables are present
        missing = [obs for obs in self.required_observables if obs not in observables]
        if missing:
            return ObjectiveOutput(
                is_ready=False,
                needs_update=tuple(missing),
            )

        # Extract trajectories from observables
        sorted_obs = [observables[key] for key in self.required_observables]
        trajectories = [obs for obs in sorted_obs if isinstance(obs, SimulatorTrajectory)]

        if not trajectories:
            raise ValueError("No SimulatorTrajectory observables found in observables.")

        if self.n_equilibration_steps > 0:

            def slc_f(n: int) -> slice:
                return slice(self.n_equilibration_steps, n, None)

            trajectories = [obs.slice(slc_f(obs.length())) for obs in trajectories]

        reference_states = SimulatorTrajectory.concat(trajectories)
        if reference_states.length() == 0:
            raise ValueError(
                "Equilibration slicing yields no states! Note slicing is in number of snapshots, not timesteps."
            )

        # Derive beta from trajectory temperature
        if reference_states.temperature is None:
            raise ValueError(
                "SimulatorTrajectory.temperature is None. "
                "DiffTRe requires per-state temperature (kT) on the trajectory."
            )
        beta = 1.0 / reference_states.temperature

        # The reference opt params will be returned in state while we are still
        # within neff tolerance (or max_valid_opt_steps). These params are then
        # used to compute reference energies.
        reference_opt_params = reference_opt_params or opt_params
        reference_energies = self.energy_fn.with_params(reference_opt_params).map(reference_states)

        # Compute per-segment neff to check if trajectory is still valid
        neff = compute_min_segment_neff(
            temperature=reference_states.temperature,
            new_energies=self.energy_fn.with_params(opt_params).map(reference_states),
            ref_energies=reference_energies,
        )

        # check if trajectory needs recomputation due to low effective sample size
        if neff < self.min_n_eff_factor:
            return ObjectiveOutput(
                is_ready=False,
                needs_update=tuple(self.required_observables),
                observables={"neff": neff},
                state={"opt_steps": 0},
            )

        # Compute gradients
        (loss, (_, measured_value, new_energies)), grads = compute_loss_and_grad(
            opt_params,
            self.energy_fn,
            beta,
            self.grad_or_loss_fn,
            reference_states,
            reference_energies,
            sorted_obs,
        )

        # Build output observables
        output_observables = {
            "loss": loss,
            "neff": neff,
            measured_value[0]: measured_value[1],
        }

        return ObjectiveOutput(
            is_ready=True,
            grads=grads,
            observables=output_observables,
            state={
                "opt_steps": opt_steps + 1,
                "reference_opt_params": reference_opt_params,
            },
        )
