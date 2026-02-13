"""Tests for Martini 3 angle potential energy function."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax_md import space

from mythos.energy.martini.base import MartiniTopology
from mythos.energy.martini.m2.angle import AngleConfiguration, triplet_angle
from mythos.energy.martini.m3.angle import Angle
from mythos.simulators.gromacs.utils import read_trajectory_mdanalysis
from mythos.simulators.io import SimulatorTrajectory

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - common jax practice

# Test data directory with GROMACS trajectory files
TEST_DATA_DIR = Path("data/test-data/martini/energy/m3/angle")


def load_angle_params() -> dict:
    """Load angle parameters from the JSON configuration file.

    Note: Unlike M2 params, M3 params are already in radians.
    """
    with (TEST_DATA_DIR / "angle_params_rad.json").open() as f:
        return json.load(f)


@pytest.fixture
def gromacs_trajectory() -> SimulatorTrajectory:
    """Load the test trajectory from GROMACS output files."""
    return read_trajectory_mdanalysis(
        TEST_DATA_DIR / "test.tpr",
        TEST_DATA_DIR / "test.trr",
    )


@pytest.fixture
def angle_config() -> AngleConfiguration:
    """Create an AngleConfiguration from the JSON parameters file."""
    params = load_angle_params()
    return AngleConfiguration(**params)


@pytest.fixture
def energies() -> jnp.ndarray:
    """Load reference angle energies from GROMACS output file."""
    with (TEST_DATA_DIR / "angle.xvg").open() as f:
        energy_values = []
        for line in f:
            if not line.startswith(("#", "@")):
                _, energy = line.strip().split()
                energy_values.append(float(energy))
    return jnp.array(energy_values[1:])


class TestAngleEnergy:
    """Tests for Martini 3 Angle energy computation."""

    def test_angle_energy_against_gromacs(
        self,
        gromacs_trajectory: SimulatorTrajectory,
        angle_config: AngleConfiguration,
        energies: jnp.ndarray,
    ):
        """Test angle energy calculation matches GROMACS reference values."""
        top = MartiniTopology.from_tpr(TEST_DATA_DIR / "test.tpr", unbonded=jnp.array([]))

        angle_fn = Angle.from_topology(
            topology=top,
            params=angle_config,
        )

        computed_energies = angle_fn.map(gromacs_trajectory)
        assert energies.shape[0] == gromacs_trajectory.length()
        assert jnp.allclose(computed_energies, energies)


class TestUseG96:
    """Test that Martini 3 uses harmonic (use_G96=False) angle potential.

    With use_G96=False the potential is V = 0.5 * k * (theta - theta0)^2,
    unlike Martini 2 which uses V = 0.5 * k * (cos(theta) - cos(theta0))^2.
    """

    def test_harmonic_angle_potential(self):
        """Verify use_G96=False (from m3 Angle) produces the standard harmonic potential."""
        k = 25.0
        theta0 = jnp.pi / 2
        deviation = jnp.deg2rad(20.0)
        angle = theta0 - deviation

        centers = jnp.array([
            [jnp.cos(angle), jnp.sin(angle), 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        triplet = jnp.array([0, 1, 2])
        displacement_fn, _ = space.free()

        energy = triplet_angle(centers, triplet, k, theta0, displacement_fn, use_G96=Angle.use_G96)

        # Should match 0.5 * k * (theta - theta0)^2
        expected = 0.5 * k * deviation ** 2
        assert jnp.isclose(energy, expected, atol=1e-8)
