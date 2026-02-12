"""Tests for the MembraneThickness observable."""

from pathlib import Path

import jax
import MDAnalysis
import numpy as np
import pytest

from mythos.observables.membrane_thickness import MembraneThickness
from mythos.simulators.gromacs.utils import read_trajectory_mdanalysis
from mythos.simulators.io import SimulatorTrajectory

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - common jax practice

# For these tests we recycle the trajectory from LennardJones tests, it just
# needs to be some system where we've already calculated the thickness.
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


class TestMembraneThickness:
    """Tests for MembraneThickness observable."""

    def test_thickness_against_expected(
        self,
        topology: MDAnalysis.Universe,
        gromacs_trajectory: SimulatorTrajectory,
    ):
        """Test membrane thickness calculation against known reference values."""
        expected = np.array([
            37.21121013, 36.94640994, 37.31411836, 37.03461868, 36.75582552,
            36.76741627, 37.21104291, 36.92698368, 36.80011913, 36.98599377,
        ])

        observable = MembraneThickness(
            topology=topology,
            lipid_sel="name GL1 GL2",
            thickness_sel="name PO4",
        )

        result = observable(gromacs_trajectory)
        assert result.shape == (gromacs_trajectory.length(),)
        np.testing.assert_allclose(result, expected, atol=1e-6)
