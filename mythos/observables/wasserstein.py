"""Wasserstein distance observables."""

import math
from dataclasses import field

import chex
import jax.numpy as jnp

from mythos.observables.base import BaseObservable
from mythos.simulators.io import SimulatorTrajectory
from mythos.utils.types import Arr_N, Scalar


def wasserstein_1d(u: Arr_N, v: Arr_N, u_weights: Arr_N | None = None, v_weights: Arr_N | None = None) -> Scalar:
    """Compute the 1D Wasserstein distance between two distributions u and v."""
    u = jnp.asarray(u, dtype=jnp.float64)
    v = jnp.asarray(v, dtype=jnp.float64)

    if u_weights is None:
        u_weights = jnp.full(u.shape, 1.0 / u.size, dtype=jnp.float64)
    else:
        u_weights = jnp.asarray(u_weights, dtype=jnp.float64)

    if v_weights is None:
        v_weights = jnp.full(v.shape, 1.0 / v.size, dtype=jnp.float64)
    else:
        v_weights = jnp.asarray(v_weights, dtype=jnp.float64)

    if u_weights.shape != u.shape:
        raise ValueError(f"u_weights must have the same shape as u; got {u_weights.shape} and {u.shape}.")

    if v_weights.shape != v.shape:
        raise ValueError(f"v_weights must have the same shape as v; got {v_weights.shape} and {v.shape}.")

    # Validate that total masses match (within numerical tolerance)
    if not jnp.isclose(jnp.sum(u_weights), jnp.sum(v_weights), rtol=1e-5, atol=1e-5):
        raise ValueError(
            "u_weights and v_weights must sum to the same total mass; "
            f"got {jnp.sum(u_weights)} and {jnp.sum(v_weights)}."
        )

    # Sort u and v with their weights
    u_sort_idx = jnp.argsort(u)
    v_sort_idx = jnp.argsort(v)

    u = u[u_sort_idx]
    v = v[v_sort_idx]
    u_weights = u_weights[u_sort_idx]
    v_weights = v_weights[v_sort_idx]

    # Merge all support points
    all_vals = jnp.concatenate([u, v])
    all_weights = jnp.concatenate([u_weights, -v_weights])
    sort_idx = jnp.argsort(all_vals)
    all_vals = all_vals[sort_idx]
    all_weights = all_weights[sort_idx]

    # Compute CDF difference over each interval
    diffs = jnp.cumsum(all_weights)
    dx = all_vals[1:] - all_vals[:-1]
    avg_heights = jnp.abs(diffs[:-1])

    return jnp.sum(dx * avg_heights)


def _compute_wasserstein_distance(
    obs_values: Arr_N, v: Arr_N, weights: Arr_N | None = None, v_weights: Arr_N | None = None
) -> Scalar:
    obs_shape = obs_values.shape
    # flatten the observable output if it's not already 1D
    obs_values = obs_values.flatten()
    # reshape weights to match flattened obs_values, if provided. Each weight is
    # expected to correspond to a state in trajectory, thus we need to copy
    # those into the per-state distribution of obs_values.
    if weights is not None:
        n_per_weight = math.prod(obs_shape[1:], start=1)
        weights = jnp.repeat(weights, n_per_weight) / n_per_weight
    return wasserstein_1d(obs_values, v, u_weights=weights, v_weights=v_weights)


@chex.dataclass(frozen=True, kw_only=True)
class WassersteinDistance:
    """Compute the 1D Wasserstein distance between two distributions.

    The U distribution is obtained by calling the supplied observable on the
    trajectory, and the V distribution is provided as a fixed reference
    distribution. Weights can optionally be provided for the V distribution as a
    property, and for the U distribution at call time.

    The observable, when called on a trajectory, should return a (n_states,
    n_values) array, where n_states is the number of states in the trajectory.
    This will be flattened on its way into the Wasserstein distance computation.

    The weights supplied to the call method are expected to correspond to states
    in the trajectory, and will apply to all values in the observable output
    distribution for that state.

    Attributes:
        observable: The observable whose output distribution defines U.
        v_distribution: The fixed reference distribution V to compare against.
        v_weights: Optional weights for the V distribution (should sum to 1).
    """

    observable: BaseObservable
    v_distribution: Arr_N
    v_weights: Arr_N | None = None

    def __call__(self, trajectory: SimulatorTrajectory, weights: Arr_N | None = None) -> Scalar:
        """Compute the Wasserstein distance between observable and reference distributions."""
        obs_values = self.observable(trajectory)
        return _compute_wasserstein_distance(obs_values, self.v_distribution, weights=weights, v_weights=self.v_weights)


@chex.dataclass(frozen=True, kw_only=True)
class WassersteinDistanceMapped:
    """Compute the 1D Wasserstein distance between two distributions, by key.

    This is a generalization of WassersteinDistance that allows computing
    distances for multiple observables and reference distributions at once, by
    key. The input observable is expected to return a dictionary mapping keys to
    observable outputs, and the v_distribution_map (value corresponding to
    v_distribution of `:class:WassersteinDistanceMapped`) and v_weights_map
    (value corresponding to v_weights of `:class:WassersteinDistanceMapped`)
    should have matching keys.

    See `:class:WassersteinDistance` for more information on inputs and calling.

    Attributes:
        observable: The observable whose output distribution defines U, expected
            to return a dictionary mapping keys to observable outputs.
        v_distribution_map: Dictionary mapping keys to fixed reference
            distributions V to compare against.
        v_weights_map: Optional dictionary mapping keys to weights for the V
            distributions.
    """

    observable: BaseObservable
    v_distribution_map: dict[str, Arr_N]
    v_weights_map: dict[str, Arr_N | None] = field(default_factory=dict)

    def __call__(self, trajectory: SimulatorTrajectory, weights: Arr_N | None = None) -> dict[str, Scalar]:
        """Compute the Wasserstein distance between all observable and reference distributions, by key."""
        obs_values = self.observable(trajectory)
        return {
            key: _compute_wasserstein_distance(
                obs_values[key], self.v_distribution_map[key], weights=weights, v_weights=self.v_weights_map.get(key)
            )
            for key in self.v_distribution_map
        }
