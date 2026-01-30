"""Bond potential energy function for Martini 2."""
import chex
import jax
import jax.numpy as jnp
from jax_md import space
from typing_extensions import override

from mythos.energy.martini.base import MartiniEnergyConfiguration, MartiniEnergyFunction
from mythos.simulators.io import SimulatorTrajectory
from mythos.utils.types import Arr_States_3, Vector2D

BOND_K_PREFIX = "bond_k_"
BOND_R0_PREFIX = "bond_r0_"

class BondConfiguration(MartiniEnergyConfiguration):
    """Configuration for Martini bond energy function.

    Bond params must be provided as "bond_k_NAME" and "bond_r0_NAME" in
    corresponding pairs for each bond name in the system. NAME should be in the
    format of "MOLTYPE_ATOMNAME1_ATOMNAME2", e.g., "DMPC_NC3_PO4".
    """

    @override
    def __post_init__(self) -> None:
        for param in self.params:
            if not param.startswith((BOND_K_PREFIX, BOND_R0_PREFIX)):
                raise ValueError(f"Unexpected parameter {param} for BondConfiguration")
        if len(self.params) == 0 or len(self.params) % 2 != 0:
            raise ValueError("BondConfiguration requires pairs of k and r0 parameters")


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
    def __post_init__(self, topology: None = None) -> None:
        # cache parameters mapped to bonds by indices. The result is arrays of
        # len(bonded_neighbors) where each element corresponds to the k or r0 for that bond.
        k = [self.params[BOND_K_PREFIX + name] for name in self.bond_names]
        r0 = [self.params[BOND_R0_PREFIX + name] for name in self.bond_names]
        object.__setattr__(self, "_bonds_k", jnp.array(k))
        object.__setattr__(self, "_bonds_r0", jnp.array(r0))

    @override
    def compute_energy(self, trajectory: SimulatorTrajectory) -> float:
        displacement_fn = self.displacement_fn(trajectory.box_size)
        # Using our cached per-bond parameters, we map over the triplicate of
        # bond pairs, k values, and r0 values.
        pair_vmap = jax.vmap(pair_bond, in_axes=(None, 0, 0, 0, None))
        return pair_vmap(
            trajectory.center,
            self.bonded_neighbors,
            self._bonds_k,
            self._bonds_r0,
            displacement_fn,
        ).sum()
