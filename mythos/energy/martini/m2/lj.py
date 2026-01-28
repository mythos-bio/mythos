"""Lennard-Jones potential energy function for Martini 2."""

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jax_md import space
from typing_extensions import override

from mythos.energy.configuration import BaseConfiguration
from mythos.energy.martini.base import MartiniEnergyFunction, get_periodic
from mythos.simulators.io import SimulatorTrajectory
from mythos.utils.types import Arr_N, Arr_States_3, MatrixSq, Vector2D


@chex.dataclass(frozen=True, kw_only=True)
class LJConfiguration(BaseConfiguration):
    """Configuration for Lennard-Jones potential in Martini 2."""
    bead_types: tuple[str, ...]  # Bead type corresponding to sigma/epsilon indices
    sigmas: MatrixSq  # shape: (n_types, n_types)
    epsilons: MatrixSq  # shape: (n_types, n_types)

    required_params = ("sigmas", "epsilons")
    non_optimizable_required_params = ("bead_types",)

    @override
    def __post_init__(self) -> None:
        bead_type_shape = (len(self.bead_types), len(self.bead_types))
        if not (bead_type_shape == self.sigmas.shape == self.epsilons.shape):
            raise ValueError("sigmas and epsilons must have shape (n_types, n_types)")


def lennard_jones(r: float, eps: float, sigma: float) -> float:
    """Calculate Lennard-Jones potential given distance r, epsilon, and sigma."""
    cutoff = 1.1
    # calculating the standard LJ potential
    v = 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
    # calculating the value of the potential at cutoff
    v_c = 4 * eps * ((sigma / cutoff) ** 12 - (sigma / cutoff) ** 6)
    # applying the shifting function: V_s(r) = V(r) - V(r_c) for r < r_c, 0 otherwise
    return jnp.where(
        r < cutoff, v - v_c, 0.0  # shifting the potential by subtracting V(r_c)
    )

def pair_lj(
        centers: Arr_States_3,
        pair: Vector2D,
        sigmas: MatrixSq,
        epsilons: MatrixSq,
        types: Arr_N,
        displacement_fn: callable,
    ) -> float:
    """Calculate LJ energy for a given pair of particles."""
    i = pair[0]
    j = pair[1]

    i_type = types[i]
    j_type = types[j]

    sigma = sigmas[i_type, j_type]
    eps = epsilons[i_type, j_type]

    r = space.distance(displacement_fn(centers[i], centers[j]))
    return lennard_jones(r, eps, sigma)

@chex.dataclass(frozen=True, kw_only=True)
class LJ(MartiniEnergyFunction):
    """Lennard-Jones potential energy function for Martini 2."""

    params: LJConfiguration
    displacement_fn: Callable = get_periodic

    @override
    def __post_init__(self, topology: None = None) -> None:
        # Cache a mapping between atom index and its type within sigma/epsilon
        # matrices
        type_map = {t: i for i,t in enumerate(self.params.bead_types)}
        atom_type_map = jnp.array([type_map[t] for t in self.atom_types])
        object.__setattr__(self, "_atom_type_map", atom_type_map)

    @override
    def compute_energy(self, trajectory: SimulatorTrajectory) -> float:
        displacement_fn = self.displacement_fn(trajectory.box_size / 10.0)
        ljmap = jax.vmap(pair_lj, in_axes=(None, 0, None, None, None, None))
        return ljmap(
            trajectory.center / 10.0,
            self.unbonded_neighbors,
            self.params.sigmas,
            self.params.epsilons,
            self._atom_type_map,
            displacement_fn,
        ).sum()
