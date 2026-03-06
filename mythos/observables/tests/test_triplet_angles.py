"""Tests for the TripletAngles observable."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mythos.energy.martini.base import MartiniTopology
from mythos.observables.triplet_angles import TripletAngles
from mythos.simulators.io import SimulatorTrajectory

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - common jax practice


def _make_topology(
    atom_names: tuple[str, ...],
    residue_names: tuple[str, ...],
    angles: jnp.ndarray,
) -> MartiniTopology:
    """Create a minimal MartiniTopology for testing."""
    return MartiniTopology(
        atom_types=atom_names,
        atom_names=atom_names,
        residue_names=residue_names,
        angles=angles,
        bonded_neighbors=jnp.zeros((0, 2), dtype=jnp.int32),
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


class TestTripletAngles:
    """Tests for the TripletAngles observable."""

    def test_right_angle(self):
        """Three atoms at a 90-degree angle should return pi/2."""
        topology = _make_topology(
            atom_names=("A", "B", "C"),
            residue_names=("MOL", "MOL", "MOL"),
            angles=jnp.array([[0, 1, 2]]),
        )
        # i=(1,0,0), j=(0,0,0), k=(0,1,0) → 90 degrees at j
        centers = jnp.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        box_size = jnp.array([[20.0, 20.0, 20.0]])
        traj = _make_trajectory(centers, box_size)

        obs = TripletAngles(topology=topology, angle_names=("MOL_A_B_C",))
        result = obs(traj)

        assert "MOL_A_B_C" in result
        assert result["MOL_A_B_C"].shape == (1, 1)
        np.testing.assert_allclose(result["MOL_A_B_C"][0, 0], jnp.pi / 2, atol=1e-8)

    def test_straight_angle(self):
        """Three collinear atoms (i-j-k opposite sides) should return pi."""
        topology = _make_topology(
            atom_names=("A", "B", "C"),
            residue_names=("MOL", "MOL", "MOL"),
            angles=jnp.array([[0, 1, 2]]),
        )
        # i=(1,0,0), j=(0,0,0), k=(-1,0,0) → 180 degrees
        centers = jnp.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
        box_size = jnp.array([[20.0, 20.0, 20.0]])
        traj = _make_trajectory(centers, box_size)

        obs = TripletAngles(topology=topology, angle_names=("MOL_A_B_C",))
        result = obs(traj)

        np.testing.assert_allclose(result["MOL_A_B_C"][0, 0], jnp.pi, atol=1e-8)

    def test_60_degree_angle(self):
        """Three atoms at a 60-degree angle."""
        topology = _make_topology(
            atom_names=("A", "B", "C"),
            residue_names=("MOL", "MOL", "MOL"),
            angles=jnp.array([[0, 1, 2]]),
        )
        # i at (1,0,0), j at origin, k at 60 degrees from i
        centers = jnp.array([
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, jnp.sqrt(3.0) / 2, 0.0]]
        ])
        box_size = jnp.array([[20.0, 20.0, 20.0]])
        traj = _make_trajectory(centers, box_size)

        obs = TripletAngles(topology=topology, angle_names=("MOL_A_B_C",))
        result = obs(traj)

        np.testing.assert_allclose(result["MOL_A_B_C"][0, 0], jnp.pi / 3, atol=1e-8)

    def test_multiple_states(self):
        """Angles change across multiple trajectory states."""
        topology = _make_topology(
            atom_names=("A", "B", "C"),
            residue_names=("MOL", "MOL", "MOL"),
            angles=jnp.array([[0, 1, 2]]),
        )
        # State 0: 90 degrees, State 1: 180 degrees
        centers = jnp.array([
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        ])
        box_size = jnp.array([
            [20.0, 20.0, 20.0],
            [20.0, 20.0, 20.0],
        ])
        traj = _make_trajectory(centers, box_size)

        obs = TripletAngles(topology=topology, angle_names=("MOL_A_B_C",))
        result = obs(traj)

        assert result["MOL_A_B_C"].shape == (2, 1)
        np.testing.assert_allclose(
            result["MOL_A_B_C"].flatten(), [jnp.pi / 2, jnp.pi], atol=1e-8
        )

    def test_multiple_matching_angles(self):
        """Two angles with the same name appear in a single array."""
        topology = _make_topology(
            atom_names=("A", "B", "C", "A", "B", "C"),
            residue_names=("MOL", "MOL", "MOL", "MOL", "MOL", "MOL"),
            angles=jnp.array([[0, 1, 2], [3, 4, 5]]),
        )
        # First triplet: 90 degrees, Second triplet: 180 degrees
        centers = jnp.array([
            [
                [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],    # 90°
                [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0],   # 180°
            ],
        ])
        box_size = jnp.array([[20.0, 20.0, 20.0]])
        traj = _make_trajectory(centers, box_size)

        obs = TripletAngles(topology=topology, angle_names=("MOL_A_B_C",))
        result = obs(traj)

        assert result["MOL_A_B_C"].shape == (1, 2)
        np.testing.assert_allclose(
            result["MOL_A_B_C"][0], [jnp.pi / 2, jnp.pi], atol=1e-8
        )

    def test_multiple_angle_names(self):
        """Requesting multiple distinct angle names returns a dict with both."""
        topology = _make_topology(
            atom_names=("A", "B", "C", "D", "E", "F"),
            residue_names=("R1", "R1", "R1", "R2", "R2", "R2"),
            angles=jnp.array([[0, 1, 2], [3, 4, 5]]),
        )
        # R1: 90 degrees, R2: 180 degrees
        centers = jnp.array([
            [
                [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            ],
        ])
        box_size = jnp.array([[20.0, 20.0, 20.0]])
        traj = _make_trajectory(centers, box_size)

        obs = TripletAngles(
            topology=topology,
            angle_names=("R1_A_B_C", "R2_D_E_F"),
        )
        result = obs(traj)

        assert set(result.keys()) == {"R1_A_B_C", "R2_D_E_F"}
        np.testing.assert_allclose(result["R1_A_B_C"][0, 0], jnp.pi / 2, atol=1e-8)
        np.testing.assert_allclose(result["R2_D_E_F"][0, 0], jnp.pi, atol=1e-8)

    def test_unknown_angle_name_raises(self):
        """Requesting an angle name not in the topology raises ValueError."""
        topology = _make_topology(
            atom_names=("A", "B", "C"),
            residue_names=("MOL", "MOL", "MOL"),
            angles=jnp.array([[0, 1, 2]]),
        )
        obs = TripletAngles(topology=topology, angle_names=("NONEXISTENT",))

        centers = jnp.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        box_size = jnp.array([[20.0, 20.0, 20.0]])
        traj = _make_trajectory(centers, box_size)

        with pytest.raises(ValueError, match="No angles matching 'NONEXISTENT'"):
            obs(traj)

    def test_periodic_boundary_wrapping(self):
        """Angle computation respects periodic boundary conditions."""
        topology = _make_topology(
            atom_names=("A", "B", "C"),
            residue_names=("MOL", "MOL", "MOL"),
            angles=jnp.array([[0, 1, 2]]),
        )
        # Box of size 10; atom i is near the box edge, should wrap correctly
        # j at (5,5,5), i at (9,5,5) → displacement (4,0,0)
        # k at (5,9,5) → displacement (0,4,0) → 90 degree angle
        centers = jnp.array([[[9.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 9.0, 5.0]]])
        box_size = jnp.array([[10.0, 10.0, 10.0]])
        traj = _make_trajectory(centers, box_size)

        obs = TripletAngles(topology=topology, angle_names=("MOL_A_B_C",))
        result = obs(traj)

        np.testing.assert_allclose(result["MOL_A_B_C"][0, 0], jnp.pi / 2, atol=1e-8)
