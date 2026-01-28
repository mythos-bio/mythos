"""Bond potential energy function for Martini 2."""
import chex
import jax
import jax.numpy as jnp
from jax_md import space
from typing_extensions import override

from mythos.energy.configuration import BaseConfiguration
from mythos.energy.martini.base import MartiniEnergyFunction
from mythos.simulators.io import SimulatorTrajectory
from mythos.utils.types import Arr_Bonded_Neighbors, Arr_N, Arr_States_3, Vector2D


@chex.dataclass(frozen=True, kw_only=True)
class BondConfiguration(BaseConfiguration):
    """Configuration for Martini bond energy function."""

    bond_names: tuple[str, ...]
    k: Arr_N
    r0: Arr_N

    required_params = ("k", "r0")
    non_optimizable_required_params = ("bond_names",)


def pair_bond(
        centers: Arr_States_3,
        pair: Vector2D,
        k_bond: float,
        r0_bond: float,
        displacement_fn: callable
    ) -> float:
    """Calculate bond energy for a given pair of particles."""
    i = pair[0]
    j = pair[1]

    r = space.distance(displacement_fn(centers[i], centers[j]))
    return 0.5 * k_bond * (r - r0_bond) ** 2


@chex.dataclass(frozen=True, kw_only=True)
class Bond(MartiniEnergyFunction):
    """Bond potential energy function for Martini 2."""

    params: BondConfiguration

    @override
    def __post_init__(self) -> None:
        # cache parameters mapped to bonds by indices
        bond_name_to_index = {name: idx for idx, name in enumerate(self.params.bond_names)}
        bond_index_map = jnp.array([bond_name_to_index[name] for name in self.bond_names])
        bonds_k = self.params.k[bond_index_map]
        bonds_r0 = self.params.r0[bond_index_map]
        object.__setattr__(self, "_bonds_k", bonds_k)
        object.__setattr__(self, "_bonds_r0", bonds_r0)

    @override
    def compute_energy(self, trajectory: SimulatorTrajectory) -> float:
        displacement_fn = self.displacement_fn(trajectory.box_size)
        pair_vmap = jax.vmap(pair_bond, in_axes=(None, 0, 0, 0, None))
        return pair_vmap(
            trajectory.center,
            self.bonded_neighbors,
            self._bonds_k,
            self._bonds_r0,
            displacement_fn,
        ).sum()
