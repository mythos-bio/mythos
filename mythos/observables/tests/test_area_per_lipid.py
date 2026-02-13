"""Tests for the AreaPerLipid observable."""

from pathlib import Path

import jax
import MDAnalysis
import numpy as np
import pytest

from mythos.observables.area_per_lipid import AreaPerLipid
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


class TestAreaPerLipid:
    """Tests for AreaPerLipid observable."""

    def test_thickness_against_expected(
        self,
        topology: MDAnalysis.Universe,
        gromacs_trajectory: SimulatorTrajectory,
    ):
        """Test membrane thickness calculation against known reference values."""
        expected = np.array([
            51.189245, 51.382128, 50.695458, 51.42874 , 51.178519,
            51.148737, 50.517493, 51.3376  , 51.586332, 51.005933
        ])

        observable = AreaPerLipid(
            topology=topology,
            lipid_sel="name GL1 GL2",
        )

        result = observable(gromacs_trajectory)
        assert result.shape == (gromacs_trajectory.length(),)
        np.testing.assert_allclose(result, expected, atol=1e-6)
