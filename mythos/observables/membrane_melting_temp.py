r"""Membrane melting temperature observable.

Computes the melting temperature (Tm) of a lipid membrane by fitting a sigmoid
to area-per-lipid (APL) vs. temperature data, following the approach from
jax-martini (Pastor et al.).  The sigmoid model is:

.. math::
    \text{APL}(T) = \text{apl}_0 + c_{pg} \cdot T
        + \frac{\Delta\text{APL}}{1 + \exp(-k (T - T_m))}

The five fit parameters are ``[apl0, c_p_g, dAPL, k, Tm]``.

The module provides both standalone functions for sigmoid fitting and a
:class:`MembraneMeltingTemp` observable class that takes a
:class:`~mythos.simulators.io.SimulatorTrajectory` as input.
"""

import chex
import jax.numpy as jnp
import jax.ops
import MDAnalysis
from jaxopt import LevenbergMarquardt

from mythos.observables.area_per_lipid import AreaPerLipid
from mythos.simulators.io import SimulatorTrajectory


def calculate_apl(
    t: jnp.ndarray,
    apl0: float,
    c_p_g: float,
    dAPL: float,  # noqa: N803 — matches jax-martini naming
    k: float,
    Tm: float,  # noqa: N803 — matches jax-martini naming
) -> jnp.ndarray:
    """Evaluate the APL sigmoid model at temperature(s) *t*.

    Args:
        t: Temperature(s) in Kelvin.
        apl0: Baseline APL (gel phase).
        c_p_g: Linear temperature coefficient.
        dAPL: APL jump across the transition.
        k: Steepness of the sigmoid.
        Tm: Melting temperature.

    Returns:
        Predicted APL value(s).
    """
    return apl0 + c_p_g * t + dAPL / (1 + jnp.exp(-k * (t - Tm)))


def apl_residual(
    coeffs: jnp.ndarray,
    data: tuple[jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Residual function for least-squares sigmoid fitting.

    Follows the ``residual_fun(params, *args)`` convention expected
    by :class:`jaxopt.LevenbergMarquardt`.  The data arguments are packed
    into a single tuple to ensure compatibility with jaxopt's implicit
    differentiation.

    Args:
        coeffs: Parameter vector ``[apl0, c_p_g, dAPL, k, Tm]``.
        data: Tuple of ``(sim_apls, sim_temps)`` where *sim_apls* are the
            observed APL values and *sim_temps* the corresponding
            temperatures, both of shape ``(n_temps,)``.

    Returns:
        Element-wise residual ``sim_apls - predicted_apls``.
    """
    sim_apls, sim_temps = data
    apl0, c_p_g, dAPL, k, Tm = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]  # noqa: N806
    return sim_apls - calculate_apl(sim_temps, apl0, c_p_g, dAPL, k, Tm)


def get_initial_guess(sim_apls: jnp.ndarray, sim_temps: jnp.ndarray) -> jnp.ndarray:
    """Heuristic initial guess for the sigmoid parameters.

    Args:
        sim_apls: Observed APL values, shape ``(n_temps,)``.
        sim_temps: Corresponding temperatures, shape ``(n_temps,)``.

    Returns:
        Parameter vector ``[apl0, c_p_g, dAPL, k, Tm]``.
    """
    apl0 = jnp.min(sim_apls) - 0.0001 * 276
    c_p_g = 1e-4
    dAPL = jnp.max(sim_apls) - jnp.min(sim_apls)  # noqa: N806
    k = 1.0
    Tm = jnp.median(sim_temps)  # noqa: N806
    return jnp.array([apl0, c_p_g, dAPL, k, Tm])


def fit_apl_sigmoid(
    sim_apls: jnp.ndarray,
    sim_temps: jnp.ndarray,
    *,
    implicit_diff: bool = True,
    maxiter: int = 5000,
) -> jnp.ndarray:
    """Fit the sigmoid model to APL-vs-temperature data via nonlinear least squares.

    Uses Levenberg-Marquardt, which is more robust than Gauss-Newton for the
    strongly nonlinear sigmoid model.

    Args:
        sim_apls: Observed (or reweighted) APL values, shape ``(n_temps,)``.
        sim_temps: Corresponding temperatures in Kelvin, shape ``(n_temps,)``.
        implicit_diff: Whether to use implicit differentiation through the
            solver, allowing JAX to back-propagate gradients.
        maxiter: Maximum number of solver iterations.

    Returns:
        Fitted parameter vector ``[apl0, c_p_g, dAPL, k, Tm]``.
    """
    init_guess = get_initial_guess(sim_apls, sim_temps)
    lm = LevenbergMarquardt(
        residual_fun=apl_residual, implicit_diff=implicit_diff, maxiter=maxiter,
    )
    res = lm.run(init_guess, (sim_apls, sim_temps))
    return res.params


def compute_membrane_tm(
    sim_apls: jnp.ndarray,
    sim_temps: jnp.ndarray,
    *,
    implicit_diff: bool = True,
) -> float:
    """Compute the membrane melting temperature from APL-vs-temperature data.

    Convenience wrapper around :func:`fit_apl_sigmoid` that returns just Tm.

    Args:
        sim_apls: Observed (or reweighted) APL values, shape ``(n_temps,)``.
        sim_temps: Temperatures in Kelvin, shape ``(n_temps,)``.
        implicit_diff: Whether to use implicit differentiation.

    Returns:
        Melting temperature in Kelvin.
    """
    params = fit_apl_sigmoid(sim_apls, sim_temps, implicit_diff=implicit_diff)
    return params[4]


def compute_expected_apls(
    apls: jnp.ndarray,
    weights: jnp.ndarray,
    segment_ids: jnp.ndarray,
    n_segments: int,
) -> jnp.ndarray:
    """Compute per-temperature weighted expected APL using segment sums.

    This function is fully JAX-traceable and differentiable w.r.t. *weights*.

    Args:
        apls: Per-frame APL values, shape ``(N,)``.
        weights: Per-frame importance-sampling weights, shape ``(N,)``.
        segment_ids: Integer array mapping each frame to a temperature index,
            shape ``(N,)``.  Values must be in ``[0, n_segments)``.
        n_segments: Number of distinct temperatures.

    Returns:
        Expected APL per temperature, shape ``(n_segments,)``.
    """
    weighted_apls = weights * apls
    sum_weighted_apls = jax.ops.segment_sum(weighted_apls, segment_ids, num_segments=n_segments)
    sum_weights = jax.ops.segment_sum(weights, segment_ids, num_segments=n_segments)
    return sum_weighted_apls / sum_weights


def build_segment_ids(
    temp_labels: jnp.ndarray,
    temperatures: jnp.ndarray,
    *,
    atol: float = 1.0,
) -> jnp.ndarray:
    """Map per-frame temperature labels to integer segment indices.

    For each frame, the segment id is the index into *temperatures* whose
    value is closest to the frame's temperature label.

    Args:
        temp_labels: Per-frame temperature values, shape ``(N,)``.
        temperatures: Ordered array of distinct temperatures, shape ``(T,)``.
        atol: Maximum allowed absolute difference (in Kelvin) between a
            frame's temperature label and its nearest entry in
            *temperatures*.  Raises ``ValueError`` when any frame exceeds
            this tolerance.

    Returns:
        Integer segment ids, shape ``(N,)``.

    Raises:
        ValueError: If any frame's temperature label is farther than *atol*
            from every entry in *temperatures*.
    """
    # (N, T) distance matrix; argmin along T axis gives the closest temp index
    diffs = jnp.abs(temp_labels[:, None] - temperatures[None, :])
    min_diffs = jnp.min(diffs, axis=1)
    if jnp.any(min_diffs > atol):
        raise ValueError("Trajectory temperature labels and provided temperatures do not match!")
    return jnp.argmin(diffs, axis=1)


@chex.dataclass(frozen=True, kw_only=True)
class MembraneMeltingTemp:
    """Observable that computes lipid membrane melting temperature.

    Given a concatenated :class:`SimulatorTrajectory` containing frames from
    simulations at multiple temperatures (identified via per-frame metadata),
    this observable:

    1. Computes per-frame area-per-lipid using :class:`AreaPerLipid`.
    2. Groups frames by temperature using ``trajectory.temperature``.
    3. Computes the weighted expected APL at each temperature (weighted by
       optional DiffTRe importance-sampling weights).
    4. Fits a sigmoid to APL vs. temperature and returns the melting
       temperature :math:`T_m`.

    Attributes:
        topology: MDAnalysis Universe describing the system topology.
        lipid_sel: MDAnalysis selection string for lipid tail atoms
            (e.g. ``"name GL1 GL2"``).
        temperatures: Tuple of simulation temperatures (Kelvin) to fit over.
        implicit_diff: Whether to use implicit differentiation through the
            least-squares solver.
    """

    topology: MDAnalysis.Universe
    lipid_sel: str
    temperatures: tuple[float, ...]
    implicit_diff: bool = True

    def __call__(
        self,
        trajectory: SimulatorTrajectory,
        weights: jnp.ndarray | None = None,
    ) -> float:
        """Compute the membrane melting temperature.

        Args:
            trajectory: Concatenated trajectory with per-frame temperature
                metadata under ``self.temp_key``.
            weights: Optional per-frame importance-sampling weights, shape
                ``(N,)``.  When ``None``, uniform weights are used (equivalent
                to an unweighted mean per temperature).

        Returns:
            Melting temperature in Kelvin.
        """
        apls = AreaPerLipid(topology=self.topology, lipid_sel=self.lipid_sel)(trajectory)

        # Segments match the order of input temperatures and identify the frames
        # of trajectory corresponding to each temperature.
        temps_array = jnp.array(self.temperatures)
        segment_ids = build_segment_ids(trajectory.temperature, temps_array)

        if weights is None:
            weights = jnp.ones(apls.shape[0])

        expected_apls = compute_expected_apls(apls, weights, segment_ids, len(self.temperatures))

        return compute_membrane_tm(
            expected_apls, temps_array, implicit_diff=self.implicit_diff
        )
