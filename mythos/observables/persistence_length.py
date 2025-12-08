"""Persistence length observable."""

import dataclasses as dc
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jax import vmap

import mythos.observables.base as jd_obs
import mythos.utils.types as jd_types
from mythos.simulators.io import SimulatorTrajectory

TARGETS = {
    "oxDNA": 47.5,  # nm
}


def persistence_length_fit(correlations: jnp.ndarray, l0_av: float) -> tuple[float, float]:
    """Computes the Lp given correlations in alignment decay and average distance between base pairs.

    Lp obeys the following equality: `<l_n * l_0> = exp(-n<l_0> / Lp)`, where `<l_n * l_0>` represents the
    average correlation between adjacent base pairs (`l_0`) and base pairs separated by a distance of
    `n` base pairs (`l_n`). This relationship is linear in log space, `log(<l_n * l_0>) = -n<l_0> / Lp`.
    So, given the average correlations across distances and the average distance between adjacent base pairs,
    we compute Lp via a linear fit.

    Args:
        correlations (jnp.ndarray): a (max_dist,) array containing the average correlation between
            base pairs separated by distances up to `max_dist`
        l0_av (jnp.ndarray): the average distance between adjacent base pairs
    """
    # Format the correlations for a linear fit
    y = jnp.log(correlations)
    x = jnp.arange(correlations.shape[0])
    x = jnp.stack([jnp.ones_like(x), x], axis=1)

    # Fit a line
    fit = jnp.linalg.lstsq(x, y)
    offset, slope = fit[0]
    lp = -l0_av / slope
    return lp, offset


def vector_autocorrelate(vecs: jnp.ndarray) -> jnp.ndarray:
    """Computes the average correlations in alignment decay between a list of vector.

    Given an ordered list of n vectors (representing vectors between adjacent
    base pairs), computes the average correlation between all pairs of vectors
    separated by a distance `d` for all distances `d < n`. Note that multiple
    pairs of vectors are included for all values < n-1.

    Args:
        vecs (jnp.ndarray): a (n, 3) array of vectors corresponding to
            displacements between midpoints of adjacent base pairs.

    """
    max_dist = vecs.shape[0]

    def window_correlations(i: int) -> jnp.ndarray:
        li = vecs[i]

        def i_correlation_fn(j: int) -> jnp.ndarray:
            return jnp.where(j >= i, jnp.dot(li, vecs[j]), 0.0)

        i_correlations = vmap(i_correlation_fn)(jnp.arange(max_dist))
        return jnp.roll(i_correlations, -i)

    all_correlations = vmap(window_correlations)(jnp.arange(max_dist))
    all_correlations = jnp.sum(all_correlations, axis=0)

    all_correlations /= jnp.arange(max_dist, 0, -1)
    return all_correlations


get_all_l_vectors = vmap(jd_obs.local_helical_axis_with_norm, in_axes=(0, None, None))


def compute_metadata(
    base_sites: jnp.ndarray, quartets: jnp.ndarray, displacement_fn: Callable
) -> tuple[jnp.ndarray, float]:
    """Computes (i) average correlations in alignment decay and (ii) average distance between base pairs."""
    all_l_vectors, l0_vals = get_all_l_vectors(quartets, base_sites, displacement_fn)
    autocorr = vector_autocorrelate(all_l_vectors)
    return autocorr, jnp.mean(l0_vals)


@chex.dataclass(frozen=True, kw_only=True)
class PersistenceLength(jd_obs.BaseObservable):
    """Computes the persistence length (Lp) from a trajectory.

    To model Lp, we assume an infinitely long, semi-flexible polymer, in which
    correlations in alignment decay exponentially with separation. So, to
    compute Lp, we need the average correlations across many states, as well as
    the average distance between adjacent base pairs. This observable computes
    these two quantities for a single state, and the average of these quantities
    across a trajectory can be postprocessed to compute a value for Lp.

    The callable of this class computes the weighted fitted Lp for a trajectory,
    while the `lp_fit` method computes the fitted Lp and offset. The
    `get_all_corrs_and_l0s` method computes the correlations and average
    distance between adjacent base pairs for each state in a trajectory.

    Args:
        quartets: a (n_bp-1, 2, 2) array containing the pairs of adjacent base
        pairs
            for which to compute the Lp
        displacement_fn: a function for computing displacements between two
            positions
        truncate: if provided, only consider correlations up to this distance
        skip_ends: if True, skip the first two and last two quartets when
        computing
    """

    quartets: jnp.ndarray = dc.field(hash=False)
    displacement_fn: Callable
    truncate: int | None = None
    skip_ends: bool = True

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)

    def __call__(self, trajectory: SimulatorTrajectory, weights: jnp.ndarray | None = None) -> float:
        """Calculate the fitted persistence length for a trajectory.

        Args:
            trajectory: the trajectory to calculate the persistence length for
            weights: if provided, a (n_states,) array of weights to apply to
               correlations

        Returns:
            the fitted persistence length
        """
        lp, _ = self.lp_fit(trajectory, weights)
        return lp

    def lp_fit(self, trajectory: SimulatorTrajectory, weights: jnp.ndarray | None = None) -> tuple[float, float]:
        """Calculate the fitted persistence length and offset for a trajectory.

        See arguments for `__call__`.

        Returns:
            the fitted persistence length and offset
        """
        all_corrs, all_l0s = self.get_all_corrs_and_l0s(trajectory)

        if weights is not None:
            weighted_corr_mean = jnp.dot(weights, all_corrs)
            weighted_l0_mean = jnp.dot(weights, all_l0s)
        else:
            weighted_corr_mean = jnp.mean(all_corrs, axis=0)
            weighted_l0_mean = jnp.mean(all_l0s, axis=0)

        if self.truncate:
            weighted_corr_mean = weighted_corr_mean[:self.truncate]

        fit_lp, fit_offset = persistence_length_fit(weighted_corr_mean, weighted_l0_mean)
        return fit_lp, fit_offset

    def get_all_corrs_and_l0s(self, trajectory: SimulatorTrajectory) -> tuple[jnp.ndarray, jd_types.ARR_OR_SCALAR]:
        """Calculate alignment decay and average distance correlations for adjacent base pairs.

        Args:
            trajectory: the trajectory to calculate the persistence length for

        Returns:
            tuple of (correlations, decay) the correlations in alignment decay
            and the the average distance between adjacent base pairs for each
            state. The former will have shape (n_states, n_quartets-1) and the
            latter will have shape (n_states,).
        """
        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)
        base_sites = nucleotides.base_sites

        if self.skip_ends:
            all_corrs, all_l0_vals = vmap(compute_metadata, (0, None, None))(
                base_sites[:, 2:-2, :], self.quartets[2:-2], self.displacement_fn
            )
        else:
            all_corrs, all_l0_vals = vmap(compute_metadata, (0, None, None))(
                base_sites, self.quartets, self.displacement_fn
            )

        return all_corrs, all_l0_vals
