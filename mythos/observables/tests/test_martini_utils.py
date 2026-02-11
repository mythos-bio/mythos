"""Tests for martini_utils module."""

from pathlib import Path

import jax
import MDAnalysis
import numpy as np
import pytest

from mythos.observables.martini_utils import universe_from_trajectory
from mythos.simulators.gromacs.utils import read_trajectory_mdanalysis
from mythos.simulators.io import SimulatorTrajectory

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - common jax practice

TEST_DATA_DIR = Path("data/test-data/martini/energy/m2/lj")


@pytest.fixture
def gromacs_trajectory() -> SimulatorTrajectory:
    """Load the test trajectory from GROMACS output files."""
    return read_trajectory_mdanalysis(
        TEST_DATA_DIR / "test.tpr",
        TEST_DATA_DIR / "test.trr",
    )


@pytest.fixture
def topology() -> MDAnalysis.Universe:
    """Load the topology without trajectory data."""
    return MDAnalysis.Universe(str(TEST_DATA_DIR / "test.tpr"))


@pytest.fixture
def full_universe() -> MDAnalysis.Universe:
    """Load the full universe with both topology and trajectory."""
    return MDAnalysis.Universe(
        str(TEST_DATA_DIR / "test.tpr"),
        str(TEST_DATA_DIR / "test.trr"),
    )


class TestUniverseFromTrajectory:
    """Tests for universe_from_trajectory."""

    def test_reconstructed_universe_has_correct_n_atoms(
        self,
        topology: MDAnalysis.Universe,
        gromacs_trajectory: SimulatorTrajectory,
        full_universe: MDAnalysis.Universe,
    ):
        """The reconstructed universe should have the same number of atoms as the full universe."""
        reconstructed = universe_from_trajectory(topology, gromacs_trajectory)
        assert len(reconstructed.atoms) == len(full_universe.atoms)

    def test_reconstructed_universe_has_correct_n_frames(
        self,
        topology: MDAnalysis.Universe,
        gromacs_trajectory: SimulatorTrajectory,
        full_universe: MDAnalysis.Universe,
    ):
        """The reconstructed universe should have matching frame count (minus the skipped first frame)."""
        reconstructed = universe_from_trajectory(topology, gromacs_trajectory)
        # read_trajectory_mdanalysis skips the first frame, so the reconstructed
        # trajectory has one fewer frame than the full universe.
        assert len(reconstructed.trajectory) == len(full_universe.trajectory) - 1

    def test_reconstructed_positions_match_full_universe(
        self,
        topology: MDAnalysis.Universe,
        gromacs_trajectory: SimulatorTrajectory,
        full_universe: MDAnalysis.Universe,
    ):
        """Positions in the reconstructed universe should match the full universe (after skipping frame 0)."""
        reconstructed = universe_from_trajectory(topology, gromacs_trajectory)

        for i, ts in enumerate(reconstructed.trajectory):
            # full_universe frame i+1 corresponds to reconstructed frame i
            full_universe.trajectory[i + 1]
            np.testing.assert_allclose(ts.positions, full_universe.trajectory.ts.positions, atol=1e-4)

    def test_reconstructed_box_dimensions_match_full_universe(
        self,
        topology: MDAnalysis.Universe,
        gromacs_trajectory: SimulatorTrajectory,
        full_universe: MDAnalysis.Universe,
    ):
        """Box dimensions in the reconstructed universe should match the full universe."""
        reconstructed = universe_from_trajectory(topology, gromacs_trajectory)

        for i, ts in enumerate(reconstructed.trajectory):
            full_universe.trajectory[i + 1]
            np.testing.assert_allclose(ts.dimensions, full_universe.trajectory.ts.dimensions, atol=1e-4)

    def test_reconstructed_universe_preserves_topology_attributes(
        self,
        topology: MDAnalysis.Universe,
        gromacs_trajectory: SimulatorTrajectory,
    ):
        """The reconstructed universe should preserve atom names and residue info from topology."""
        reconstructed = universe_from_trajectory(topology, gromacs_trajectory)

        np.testing.assert_array_equal(reconstructed.atoms.names, topology.atoms.names)
        np.testing.assert_array_equal(reconstructed.atoms.resnames, topology.atoms.resnames)

