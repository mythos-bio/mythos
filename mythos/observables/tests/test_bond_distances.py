"""Tests for the BondDistances and BondDistancesMapped observables."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mythos.energy.martini.base import MartiniTopology
from mythos.observables.bond_distances import BondDistances, BondDistancesMapped
from mythos.simulators.io import SimulatorTrajectory

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - common jax practice


def _make_topology(
    atom_names: tuple[str, ...],
    residue_names: tuple[str, ...],
    bonded_neighbors: jnp.ndarray,
) -> MartiniTopology:
    """Create a minimal MartiniTopology for testing."""
    return MartiniTopology(
        atom_types=atom_names,  # reuse atom_names as types for simplicity
        atom_names=atom_names,
        residue_names=residue_names,
        angles=jnp.zeros((0, 3), dtype=jnp.int32),
        bonded_neighbors=bonded_neighbors,
        unbonded_neighbors=jnp.zeros((0, 2), dtype=jnp.int32),
    )


def _make_trajectory(
    centers: jnp.ndarray,
    box_size: jnp.ndarray,
) -> SimulatorTrajectory:
    """Create a SimulatorTrajectory with given positions and box sizes."""
    n_states, n_atoms, _ = centers.shape
    orientations = jnp.zeros((n_states, n_atoms, 4))
    orientations = orientations.at[:, :, 0].set(1.0)  # identity quaternion
    return SimulatorTrajectory(
        center=centers,
        orientation=orientations,
        box_size=box_size,
    )


class TestDeriveBondNames:
    """Tests for the _derive_bond_names helper."""

    def test_names_match_topology(self):
        topology = _make_topology(
            atom_names=("NC3", "PO4", "GL1", "GL2"),
            residue_names=("DMPC", "DMPC", "DMPC", "DMPC"),
            bonded_neighbors=jnp.array([[0, 1], [1, 2], [2, 3]]),
        )
        names = topology.bond_names
        assert names == ("DMPC_NC3_PO4", "DMPC_PO4_GL1", "DMPC_GL1_GL2")

    def test_different_residues(self):
        topology = _make_topology(
            atom_names=("A", "B", "C", "D"),
            residue_names=("RES1", "RES1", "RES2", "RES2"),
            bonded_neighbors=jnp.array([[0, 1], [2, 3]]),
        )
        names = topology.bond_names
        assert names == ("RES1_A_B", "RES2_C_D")


class TestBondDistances:
    """Tests for the BondDistances observable (single bond name, returns ndarray)."""

    def test_single_bond_known_distance(self):
        """Two atoms 0.5 apart along x → distance should be 0.5."""
        topology = _make_topology(
            atom_names=("A", "B"),
            residue_names=("MOL", "MOL"),
            bonded_neighbors=jnp.array([[0, 1]]),
        )
        centers = jnp.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
        box_size = jnp.array([[10.0, 10.0, 10.0]])
        traj = _make_trajectory(centers, box_size)

        obs = BondDistances(topology=topology, bond_name="MOL_A_B")
        result = obs(traj)

        assert isinstance(result, jnp.ndarray)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result, 0.5, atol=1e-8)

    def test_multiple_states(self):
        """Distances change across multiple trajectory states."""
        topology = _make_topology(
            atom_names=("A", "B"),
            residue_names=("MOL", "MOL"),
            bonded_neighbors=jnp.array([[0, 1]]),
        )
        centers = jnp.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ])
        box_size = jnp.array([
            [20.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
        ])
        traj = _make_trajectory(centers, box_size)

        obs = BondDistances(topology=topology, bond_name="MOL_A_B")
        result = obs(traj)

        assert result.shape == (3, 1)
        np.testing.assert_allclose(result.flatten(), [1.0, 2.0, 3.0], atol=1e-8)

    def test_multiple_matching_bonds(self):
        """Two bonds with the same name should appear in a single array."""
        topology = _make_topology(
            atom_names=("A", "B", "A", "B"),
            residue_names=("MOL", "MOL", "MOL", "MOL"),
            bonded_neighbors=jnp.array([[0, 1], [2, 3]]),
        )
        centers = jnp.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ])
        box_size = jnp.array([[20.0, 20.0, 20.0]])
        traj = _make_trajectory(centers, box_size)

        obs = BondDistances(topology=topology, bond_name="MOL_A_B")
        result = obs(traj)

        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], [1.0, 2.0], atol=1e-8)

    def test_periodic_boundary_wrapping(self):
        """Distance respects periodic boundary conditions."""
        topology = _make_topology(
            atom_names=("A", "B"),
            residue_names=("MOL", "MOL"),
            bonded_neighbors=jnp.array([[0, 1]]),
        )
        centers = jnp.array([[[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])
        box_size = jnp.array([[10.0, 10.0, 10.0]])
        traj = _make_trajectory(centers, box_size)

        obs = BondDistances(topology=topology, bond_name="MOL_A_B")
        result = obs(traj)

        np.testing.assert_allclose(result[0, 0], 2.0, atol=1e-8)

    def test_unknown_bond_name_raises(self):
        """Requesting a bond name not in the topology raises ValueError."""
        topology = _make_topology(
            atom_names=("A", "B"),
            residue_names=("MOL", "MOL"),
            bonded_neighbors=jnp.array([[0, 1]]),
        )
        obs = BondDistances(topology=topology, bond_name="NONEXISTENT")

        centers = jnp.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        box_size = jnp.array([[10.0, 10.0, 10.0]])
        traj = _make_trajectory(centers, box_size)

        with pytest.raises(ValueError, match="No bonds matching 'NONEXISTENT'"):
            obs(traj)

    def test_3d_distance(self):
        """Distance computed correctly for 3D displacement."""
        topology = _make_topology(
            atom_names=("A", "B"),
            residue_names=("MOL", "MOL"),
            bonded_neighbors=jnp.array([[0, 1]]),
        )
        centers = jnp.array([[[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]]])
        box_size = jnp.array([[20.0, 20.0, 20.0]])
        traj = _make_trajectory(centers, box_size)

        obs = BondDistances(topology=topology, bond_name="MOL_A_B")
        result = obs(traj)

        np.testing.assert_allclose(result[0, 0], 3.0, atol=1e-8)


class TestBondDistancesMapped:
    """Tests for the BondDistancesMapped observable (multiple bond names, returns dict)."""

    def test_single_bond_name(self):
        """A single bond name in the tuple returns a dict with one key."""
        topology = _make_topology(
            atom_names=("A", "B"),
            residue_names=("MOL", "MOL"),
            bonded_neighbors=jnp.array([[0, 1]]),
        )
        centers = jnp.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
        box_size = jnp.array([[10.0, 10.0, 10.0]])
        traj = _make_trajectory(centers, box_size)

        obs = BondDistancesMapped(topology=topology, bond_names=("MOL_A_B",))
        result = obs(traj)

        assert isinstance(result, dict)
        assert "MOL_A_B" in result
        assert result["MOL_A_B"].shape == (1, 1)
        np.testing.assert_allclose(result["MOL_A_B"], 0.5, atol=1e-8)

    def test_multiple_bond_names(self):
        """Requesting multiple distinct bond names returns a dict with both."""
        topology = _make_topology(
            atom_names=("A", "B", "C", "D"),
            residue_names=("R1", "R1", "R2", "R2"),
            bonded_neighbors=jnp.array([[0, 1], [2, 3]]),
        )
        centers = jnp.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ])
        box_size = jnp.array([[20.0, 20.0, 20.0]])
        traj = _make_trajectory(centers, box_size)

        obs = BondDistancesMapped(
            topology=topology,
            bond_names=("R1_A_B", "R2_C_D"),
        )
        result = obs(traj)

        assert set(result.keys()) == {"R1_A_B", "R2_C_D"}
        np.testing.assert_allclose(result["R1_A_B"][0, 0], 1.0, atol=1e-8)
        np.testing.assert_allclose(result["R2_C_D"][0, 0], 3.0, atol=1e-8)
