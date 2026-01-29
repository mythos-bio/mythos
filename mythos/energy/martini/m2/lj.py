"""Lennard-Jones potential energy function for Martini 2."""


import chex
import jax
import jax.numpy as jnp
from jax_md import space
from typing_extensions import override

from mythos.energy.martini.base import MartiniEnergyConfiguration, MartiniEnergyFunction
from mythos.simulators.io import SimulatorTrajectory
from mythos.utils.types import Arr_N, Arr_States_3, MatrixSq, Vector2D


class LJConfiguration(MartiniEnergyConfiguration):
    """"Configuration for Martini Lennard-Jones energy function.

    All parameters provided must be of the form "lj_sigma_A-B" or "lj_epsilon_A-B",
    where A and B are bead types. Pair order is ignored unless both orderings
    are provided. It is required that sigma and epsilon parameters are provided
    for any bead type pairs present in the system.

    Couplings are supported (see :class:`MartiniEnergyConfiguration` for details).
    """
    @override
    def __post_init__(self) -> None:
        bead_types = set()
        for param in self.params:
            if not param.startswith(("lj_sigma_", "lj_epsilon_")):
                raise ValueError(f"Unexpected parameter {param} for LJConfiguration")
            bead_types.update(param.split("_")[2].split("-"))
        self.bead_types = tuple(sorted(bead_types))

        # Construct lookup tables for the values for use in vmapped energy
        # calculations. These should be symmetric matrices, but we do not
        # explicitly force that. At least one of the pair orderings must exist
        # or an exception is raised.
        def get_param(prefix: str, a: str, b: str) -> float:
            param = self.params.get(f"lj_{prefix}_{a}-{b}", self.params.get(f"lj_{prefix}_{b}-{a}"))
            if param is None:
                raise ValueError(f"Missing LJ {prefix} parameter for pair {a}-{b} ({b}-{a})")
            return param

        self.sigmas = jnp.array([
            [get_param("sigma", i, j) for j in self.bead_types]
            for i in self.bead_types
        ])
        self.epsilons = jnp.array([
            [get_param("epsilon", i, j) for j in self.bead_types]
            for i in self.bead_types
        ])



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

    @override
    def __post_init__(self, topology: None = None) -> None:
        # Cache a mapping between atom index and its type within sigma/epsilon
        # matrices
        type_map = {t: i for i,t in enumerate(self.params.bead_types)}
        atom_type_map = jnp.array([type_map[t] for t in self.atom_types])
        object.__setattr__(self, "_atom_type_map", atom_type_map)

    @override
    def compute_energy(self, trajectory: SimulatorTrajectory) -> float:
        displacement_fn = self.displacement_fn(trajectory.box_size)
        ljmap = jax.vmap(pair_lj, in_axes=(None, 0, None, None, None, None))
        return ljmap(
            trajectory.center,
            self.unbonded_neighbors,
            self.params.sigmas,
            self.params.epsilons,
            self._atom_type_map,
            displacement_fn,
        ).sum()
