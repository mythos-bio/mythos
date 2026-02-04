"""Angle potential energy function for Martini 2."""

from typing import ClassVar

import chex
import jax
import jax.numpy as jnp
from typing_extensions import override

from mythos.energy.martini.base import MartiniEnergyConfiguration, MartiniEnergyFunction
from mythos.simulators.io import SimulatorTrajectory
from mythos.utils.types import Arr_States_3, Vector3D

ANGLE_K_PREFIX = "angle_k_"
ANGLE_THETA0_PREFIX = "angle_theta0_"


class AngleConfiguration(MartiniEnergyConfiguration):
    """Configuration for Martini angle energy function.

    Angle params must be provided as "angle_k_I_J_K" and "angle_theta0_I_J_K" in
    corresponding pairs for each angle name in the system. NAME should be in the
    format of "MOLTYPE_ATOMNAME1_ATOMNAME2_ATOMNAME3", e.g., "DMPC_NC3_PO4_GL1".
    """

    @override
    def __post_init__(self) -> None:
        for param in self.params:
            if not param.startswith((ANGLE_K_PREFIX, ANGLE_THETA0_PREFIX)):
                raise ValueError(f"Unexpected parameter {param} for AngleConfiguration")
        if len(self.params) == 0 or len(self.params) % 2 != 0:
            raise ValueError("AngleConfiguration requires pairs of k and theta0 parameters")


def compute_angle(
        r_ij: Vector3D,
        r_kj: Vector3D,
) -> float:
    """Compute the angle between three particles (angle at j).

    Args:
        r_ij: Displacement vector from j to i.
        r_kj: Displacement vector from j to k.

    Returns:
        The angle theta_ijk in radians.
    """
    # Normalize the vectors
    r_ij_norm = r_ij / jnp.linalg.norm(r_ij)
    r_kj_norm = r_kj / jnp.linalg.norm(r_kj)


    # calculating the cross and dot products
    cross_prod = jnp.cross(r_ij_norm, r_kj_norm)
    dot_prod = jnp.dot(r_ij_norm, r_kj_norm)

    # using arctan2 for better numerical stability
    # arctan2(|a * b|, a · b) gives angle between vectors
    return jnp.arctan2(jnp.sqrt(jnp.sum(cross_prod**2)), dot_prod)


def triplet_angle(
        centers: Arr_States_3,
        triplet: Vector3D,
        k_angle: float,
        theta0_angle: float,
        displacement_fn: callable,
        use_G96: bool,  # noqa: FBT001, N803
) -> float:
    """Calculate angle energy for a given triplet of particles.

    Args:
        centers: Positions of all particles.
        triplet: Indices [i, j, k] of the three particles forming the angle.
        k_angle: Force constant for the angle.
        theta0_angle: Equilibrium angle in radians.
        displacement_fn: Function to compute displacement between particles.
        use_G96: Whether to use Gromacs 1996 cosine-based angle potential (as in
            Martini 2) or standard harmonic angle potential.

    Returns:
        Harmonic angle energy: 0.5 * k * (theta - theta0)^2
    """
    i = triplet[0]
    j = triplet[1]
    k = triplet[2]

    # Compute displacement vectors from central atom j
    r_ij = displacement_fn(centers[j], centers[i])
    r_kj = displacement_fn(centers[j], centers[k])

    theta = compute_angle(r_ij, r_kj)
    theta_term = (jnp.cos(theta) - jnp.cos(theta0_angle)) if use_G96 else (theta - theta0_angle)
    return 0.5 * k_angle * theta_term ** 2


@chex.dataclass(frozen=True, kw_only=True)
class Angle(MartiniEnergyFunction):
    """Angle potential energy function for Martini 2."""

    params: AngleConfiguration
    # https://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html#harmonicangle
    # Martini2 uses angle type 2 (G96 Angle) so MSE is defined w.r.t.
    # cos(theta). Martini3 can set this classvar to False and reuse this code.
    use_G96: ClassVar[bool] = True  # noqa: N815

    @override
    def __post_init__(self, topology: None = None) -> None:
        # Cache parameters mapped to angles by indices. The result is arrays of
        # len(angle_neighbors) where each element corresponds to the k or theta0 for that angle.
        k = [self.params[ANGLE_K_PREFIX + name] for name in self.angle_names]
        theta0 = [self.params[ANGLE_THETA0_PREFIX + name] for name in self.angle_names]
        object.__setattr__(self, "_angles_k", jnp.array(k))
        object.__setattr__(self, "_angles_theta0", jnp.array(theta0))

    @override
    def compute_energy(self, trajectory: SimulatorTrajectory) -> float:
        displacement_fn = self.displacement_fn(trajectory.box_size)
        # Using our cached per-angle parameters, we map over the triplet of
        # angle triplets, k values, and theta0 values.
        triplet_vmap = jax.vmap(triplet_angle, in_axes=(None, 0, 0, 0, None, None))
        return triplet_vmap(
            trajectory.center,
            self.angles,
            self._angles_k,
            self._angles_theta0,
            displacement_fn,
            self.use_G96,
        ).sum()
