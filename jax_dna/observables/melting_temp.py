"""Melting temperature observable."""

import dataclasses as dc
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp

import jax_dna.observables.base as jd_obs
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.types as jd_types
from jax_dna.energy import configuration
from jax_dna.utils.units import get_kt_from_c

TARGETS = {
    "SL_avg_6bp": get_kt_from_c(31.2),  # degrees
    "SL_avg_8bp": get_kt_from_c(48.2),  # degrees
    "SL_avg_12bp": get_kt_from_c(64.7),  # degrees
}


def jax_interp1d(x: jnp.ndarray, y: jnp.ndarray, x_new: float) -> jnp.ndarray:
    """Simple linear interpolation function using JAX.

    Args:
        x: Array of x coordinates
        y: Array of y coordinates
        x_new: Point(s) at which to interpolate

    Returns:
        Interpolated y value(s)
    """
    # Sort x and y if x is not already sorted
    sorted_idx = jnp.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    return jnp.interp(x_new, x_sorted, y_sorted)


def compute_finf(ratio: jnp.ndarray) -> jnp.ndarray:
    """Finite size correction to bound:unbound ratio."""
    return 1 + 1/(2*ratio) - jnp.sqrt((1 + 1/(2*ratio))**2 - 1)


def find_melting_temp(temperatures: jnp.ndarray, ratios: jnp.ndarray) -> float:
    """Find the temperature at which the concentration of single strands = 0.5 * duplex concentration.

    Args:
        temperatures: Array of temperature values
        ratios: Array of unbound:bound ratio values corresponding to each temperature
        target_ratio: The ratio value to find (default: 0.5)

    Returns:
        The interpolated temperature where ratio = target_ratio
    """
    return jax_interp1d(ratios, temperatures, 0.5)


def compute_curve_width(temperatures: jnp.ndarray, ratios: jnp.ndarray) -> float:
    """Find the width of the melting curve.

    defined as the temperature separation between unbound:bound ratio = 0.2 and
    unbound:bound ratio = 0.8

    Args:
        temperatures: Array of temperature values
        ratios: Array of unbound:bound ratio values corresponding to each temperature
    Returns:
        The width of the interpolated temperature curve between 0.2 and 0.8
    """
    return jax_interp1d(ratios, temperatures, 0.8) - jax_interp1d(ratios, temperatures, 0.2)

# has access to rigid_body_transform_fn
@chex.dataclass(frozen=True)
class MeltingTemp(jd_obs.BaseObservable):
    """Computes the melting temperature of a duplex using umbrella sampling.

    The melting temperature is defined as the temperature at which the
    concentration of DNA duplexes is double that of the concentration of single
    strands.

    Args:
        sim_temperature: float. the temperature at which the SimulatorTrajectory
            was collected, in sim. units.

        temperature_range: a vector containing the temperatures to extrapolate
            the SimulatorTrajectory data to (via histogram reweighting), in sim.
            units.

        energy_config: Energy configurations.

        energy_fn_builder: Energy function builder.
    """

    sim_temperature: float  # Temperature at which the simulation was conducted in sim. units
    temperature_range: jnp.ndarray = dc.field(hash=False)
    energy_config: list[configuration.BaseConfiguration] # needed for kt replacement in energy funcs
    energy_fn_builder: Callable[[jd_types.Params], Callable[[jnp.ndarray], jnp.ndarray]]

    def __call__(
        self,
        trajectory: jd_sio.SimulatorTrajectory,
        bind_states: jnp.ndarray,
        umbrella_weights: jnp.ndarray,
        opt_params: jd_types.PyTree,
    ) -> float:
        """Calculate the melting temperature.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory to calculate the melting temperature for
            bind_states (jnp.ndarray): an array of the sampled states of the "bond" order parameter
            umbrella_weights (jnp.ndarray): an N-dimensional array containing umbrella sampling weights
            opt_params: the parameters to optimize; use the current vals in building the energy functions

        Returns:
            float: the melting temperature in simulation units
        """
        return self.get_melting_temperature(trajectory, bind_states, umbrella_weights, opt_params)

    def get_extrap_ratios(
        self,
        trajectory: jd_sio.SimulatorTrajectory,
        bind_states: jnp.ndarray,
        umbrella_weights: jnp.ndarray,
        opt_params: jd_types.PyTree,
    ) -> float:
        """Calculate the bound:unbound ratios at the extrapolated temperatures."""
        energies_t0 = self.energy_fn_builder(opt_params)(trajectory)

        # find the unbiased ratio of bound:unbound across the temperature range
        def finf_at_t(extrapolated_temp: float) -> float:
            updates = [{"kt": extrapolated_temp} if "kt" in ec else {} for ec in self.energy_config]
            merged_params = [op | up for op, up in zip(opt_params, updates, strict=True)]

            energies_tx = self.energy_fn_builder(merged_params)(trajectory)

            boltz_factor = jnp.exp((energies_t0/self.sim_temperature) - (energies_tx/extrapolated_temp))
            unbiased_counts = (1 / umbrella_weights) * boltz_factor
            total_unbound = jnp.where(bind_states == 0, unbiased_counts, 0).sum()
            total_bound = jnp.where(bind_states != 0, unbiased_counts, 0).sum()
            phi = total_bound / total_unbound
            return compute_finf(phi) # apply finite size correction

        return jax.vmap(finf_at_t)(self.temperature_range)


    def get_melting_temperature(
        self,
        trajectory: jd_sio.SimulatorTrajectory,
        bind_states: jnp.ndarray,
        umbrella_weights: jnp.ndarray,
        opt_params: jd_types.PyTree,
    ) -> float:
        """Calculate the melting temperature."""
        extrap_ratios = self.get_extrap_ratios(trajectory, bind_states, umbrella_weights, opt_params)
        return find_melting_temp(self.temperature_range, extrap_ratios)

    def get_melting_curve(
       self,
       trajectory: jd_sio.SimulatorTrajectory,
       bind_states: jnp.ndarray,
       umbrella_weights: jnp.ndarray,
       opt_params: jd_types.PyTree,
   ) -> float:
       """Calculate the melting curve."""
       extrap_ratios = self.get_extrap_ratios(trajectory, bind_states, umbrella_weights, opt_params)
       return self.temperature_range, extrap_ratios

    def get_melting_curve_width(
        self,
        trajectory: jd_sio.SimulatorTrajectory,
        bind_states: jnp.ndarray,
        umbrella_weights: jnp.ndarray,
        opt_params: jd_types.PyTree,
    ) -> float:
        """Calculate the melting curve width."""
        extrap_ratios = self.get_extrap_ratios(trajectory, bind_states, umbrella_weights, opt_params)
        return compute_curve_width(self.temperature_range, extrap_ratios)
