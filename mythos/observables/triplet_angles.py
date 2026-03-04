"""Observable for computing angles between atom triplets from a Martini trajectory."""

import chex
import jax
import jax.numpy as jnp

from mythos.energy.martini.base import MartiniTopology, get_periodic
from mythos.energy.martini.m2.angle import compute_angle
from mythos.simulators.io import SimulatorTrajectory
from mythos.utils.types import Arr_N


def _triplet_angle(
    centers: jnp.ndarray, triplet: jnp.ndarray, displacement_fn: callable
) -> float:
    """Compute the angle (in radians) at the central atom of a triplet.

    The angle is measured at atom ``j`` between rays ``j→i`` and ``j→k``.

    Args:
        centers: Positions of all atoms, shape ``(n_atoms, 3)``.
        triplet: Array of three atom indices ``[i, j, k]``.
        displacement_fn: Displacement function respecting boundary conditions.

    Returns:
        Angle in radians.
    """
    i, j, k = triplet[0], triplet[1], triplet[2]
    r_ij = displacement_fn(centers[j], centers[i])
    r_kj = displacement_fn(centers[j], centers[k])
    return compute_angle(r_ij, r_kj)


@chex.dataclass(frozen=True, kw_only=True)
class TripletAngles:
    """Observable for computing angles between atom triplets.

    Given a :class:`MartiniTopology` and a set of angle names, this observable
    computes the angle at the central atom for all matching triplets across the
    trajectory.

    Attributes:
        topology: The Martini topology containing angle information.
        angle_names: Tuple of angle name strings to compute angles for.
            Each name has the form ``RESIDUE_BEAD1_BEAD2_BEAD3`` (e.g.
            ``"DMPC_NC3_PO4_GL1"``).  All angles in the topology matching a
            given name will be included in the output.
        displacement_fn: Factory that, given a box size vector, returns a
            displacement function respecting periodic boundary conditions.
    """

    topology: MartiniTopology
    angle_names: tuple[str, ...]
    displacement_fn: callable = get_periodic

    def _matching_triplets(self, angle_name: str) -> jnp.ndarray:
        """Return topology angle rows whose derived name matches *angle_name*.

        Returns:
            Array of shape ``(n_matching, 3)`` with atom-index triplets.

        Raises:
            ValueError: If no angles in the topology match *angle_name*.
        """
        all_names = self.topology.angle_names
        indices = [i for i, name in enumerate(all_names) if name == angle_name]
        if not indices:
            raise ValueError(
                f"No angles matching '{angle_name}' found in the topology. "
                f"Available angle names: {sorted(set(all_names))}"
            )
        return self.topology.angles[jnp.array(indices)]

    def __call__(self, trajectory: SimulatorTrajectory) -> dict[str, jnp.ndarray]:
        """Compute angles for each requested angle name.

        Args:
            trajectory: A :class:`SimulatorTrajectory` whose ``center`` has
                shape ``(n_states, n_atoms, 3)`` and ``box_size`` has shape
                ``(n_states, 3)``.

        Returns:
            Dictionary mapping each angle name to an array of angles (in
            radians) with shape ``(n_states, n_matching_angles)``.
        """
        result: dict[str, jnp.ndarray] = {}

        for angle_name in self.angle_names:
            triplets = self._matching_triplets(angle_name)  # (n_matching, 3)

            def _angles_for_state(centers: Arr_N, box: Arr_N, _triplets: Arr_N = triplets) -> Arr_N:
                disp_fn = self.displacement_fn(box)
                return jax.vmap(_triplet_angle, in_axes=(None, 0, None))(centers, _triplets, disp_fn)

            # vmap over states: (n_states, n_atoms, 3) x (n_states, 3) -> (n_states, n_matching)
            angles = jax.vmap(_angles_for_state)(trajectory.center, trajectory.box_size)
            result[angle_name] = angles

        return result
