"""Observable for computing bond distances from a Martini trajectory."""

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jax_md import space

from mythos.energy.martini.base import MartiniTopology, get_periodic
from mythos.simulators.io import SimulatorTrajectory
from mythos.utils.types import Arr_N


def _bond_distance(centers: jnp.ndarray, pair: jnp.ndarray, displacement_fn: Callable) -> float:
    """Compute the distance between a bonded pair of atoms."""
    return space.distance(displacement_fn(centers[pair[0]], centers[pair[1]]))


@chex.dataclass(frozen=True, kw_only=True)
class BondDistances:
    """Observable for computing bond distances for a single bond name.

    Given a :class:`MartiniTopology` and a bond name, this observable computes
    pairwise distances for all matching bonds across the trajectory.

    Attributes:
        topology: The Martini topology containing bond information.
        bond_name: Bond name string to compute distances for.
            Has the form ``RESIDUE_BEAD1_BEAD2`` (e.g.
            ``"DMPC_GL1_GL2"``).  All bonds in the topology matching
            this name will be included in the output.
        displacement_fn: Factory that, given a box size vector, returns a
            displacement function respecting periodic boundary conditions.
    """

    topology: MartiniTopology
    bond_name: str
    displacement_fn: Callable = get_periodic

    def _matching_pairs(self) -> jnp.ndarray:
        all_names = self.topology.bond_names
        indices = [i for i, name in enumerate(all_names) if name == self.bond_name]
        if not indices:
            raise ValueError(
                f"No bonds matching '{self.bond_name}' found in the topology. "
                f"Available bond names: {sorted(set(all_names))}"
            )
        return self.topology.bonded_neighbors[jnp.array(indices)]

    def __call__(self, trajectory: SimulatorTrajectory) -> jnp.ndarray:
        """Compute bond distances for the requested bond name.

        Args:
            trajectory: A :class:`SimulatorTrajectory` whose ``center`` has
                shape ``(n_states, n_atoms, 3)`` and ``box_size`` has shape
                ``(n_states, 3)``.

        Returns:
            Distance array of shape ``(n_states, n_matching_bonds)``.
        """
        pairs = self._matching_pairs()

        def _distances_for_state(centers: Arr_N, box: Arr_N, _pairs: Arr_N = pairs) -> Arr_N:
            disp_fn = self.displacement_fn(box)
            return jax.vmap(_bond_distance, in_axes=(None, 0, None))(centers, _pairs, disp_fn)

        # vmap over states: (n_states, n_atoms, 3) x (n_states, 3) -> (n_states, n_matching)
        return jax.vmap(_distances_for_state)(trajectory.center, trajectory.box_size)


@chex.dataclass(frozen=True, kw_only=True)
class BondDistancesMapped:
    """Observable for computing bond distances for multiple bond names.

    Given a :class:`MartiniTopology` and a set of bond names, this observable
    computes pairwise distances for all matching bonds across the trajectory,
    returning a dictionary keyed by bond name.

    Attributes:
        topology: The Martini topology containing bond information.
        bond_names: Tuple of bond name strings to compute distances for.
            Each name has the form ``RESIDUE_BEAD1_BEAD2`` (e.g.
            ``"DMPC_GL1_GL2"``).  All bonds in the topology matching a
            given name will be included in the output.
        displacement_fn: Factory that, given a box size vector, returns a
            displacement function respecting periodic boundary conditions.
    """

    topology: MartiniTopology
    bond_names: tuple[str, ...]
    displacement_fn: Callable = get_periodic

    def __call__(self, trajectory: SimulatorTrajectory) -> dict[str, jnp.ndarray]:
        """Compute bond distances for each requested bond name.

        Args:
            trajectory: A :class:`SimulatorTrajectory` whose ``center`` has
                shape ``(n_states, n_atoms, 3)`` and ``box_size`` has shape
                ``(n_states, 3)``.

        Returns:
            Dictionary mapping each bond name to a distance array of shape
            ``(n_states, n_matching_bonds)``.
        """
        return {
            bond_name: BondDistances(
                topology=self.topology,
                bond_name=bond_name,
                displacement_fn=self.displacement_fn,
            )(trajectory)
            for bond_name in self.bond_names
        }
