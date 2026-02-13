"""Tests for Martini 2 angle potential energy function."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax_md import space
from mythos.energy.martini.base import MartiniTopology
from mythos.energy.martini.m2.angle import Angle, AngleConfiguration, compute_angle, triplet_angle
from mythos.simulators.gromacs.utils import read_trajectory_mdanalysis
from mythos.simulators.io import SimulatorTrajectory

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - common jax practice

# Test data directory with GROMACS trajectory files
TEST_DATA_DIR = Path("data/test-data/martini/energy/m2/angle")
USE_G96 = True

def load_angle_params() -> dict:
    """Load angle parameters from the JSON configuration file."""
    with (TEST_DATA_DIR / "angle_params.json").open() as f:
        params = json.load(f)
    # Convert theta0 from degrees to radians
    for key in list(params.keys()):
        if key.startswith("angle_theta0_"):
            params[key] = jnp.deg2rad(params[key])
    return params


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


class TestAngleConfiguration:
    """Tests for AngleConfiguration."""

    def test_valid_configuration(self, angle_config: AngleConfiguration):
        """Test that valid configuration is created successfully from JSON."""
        assert isinstance(angle_config, AngleConfiguration)
        assert len(angle_config.params) == 12  # 6 angles with k and theta0 each

    def test_raises_error_on_bad_param_name(self):
        with pytest.raises(ValueError, match="Unexpected parameter"):
            AngleConfiguration(bad_param=100.0)

    def test_raises_error_on_odd_params(self):
        """Test that providing unpaired parameters raises an error."""
        with pytest.raises(ValueError, match="pairs of k and theta0"):
            AngleConfiguration(angle_k_A_B_C=25.0)

    def test_coupling_sets_multiple_params(self):
        """Test that a coupled parameter sets all underlying parameters."""
        couplings = {
            "angle_k_backbone": ["angle_k_A_B_C", "angle_k_D_E_F"],
        }
        params = {
            "angle_k_backbone": 25.0,
            "angle_theta0_A_B_C": 2.0,
            "angle_theta0_D_E_F": 2.5,
        }
        config = AngleConfiguration(couplings=couplings, **params)

        assert config.params["angle_k_A_B_C"] == 25.0
        assert config.params["angle_k_D_E_F"] == 25.0

    def test_coupling_opt_params_returns_coupled_name(self):
        """Test that opt_params returns coupled parameter name instead of individuals."""
        couplings = {
            "angle_k_backbone": ["angle_k_A_B_C", "angle_k_D_E_F"],
        }
        params = {
            "angle_k_backbone": 25.0,
            "angle_theta0_A_B_C": 2.0,
            "angle_theta0_D_E_F": 2.5,
        }
        config = AngleConfiguration(couplings=couplings, **params)

        opt = config.opt_params
        assert "angle_k_backbone" in opt
        assert "angle_k_A_B_C" not in opt
        assert "angle_k_D_E_F" not in opt


class TestAngleEnergy:
    """Tests for Angle energy computation."""

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


class TestComputeAngle:
    """Unit tests for the compute_angle function."""

    def test_right_angle(self):
        """Test that perpendicular vectors give pi/2 angle."""
        r_ij = jnp.array([1.0, 0.0, 0.0])
        r_kj = jnp.array([0.0, 1.0, 0.0])

        angle = compute_angle(r_ij, r_kj)
        assert jnp.isclose(angle, jnp.pi / 2, atol=1e-10)

    def test_straight_angle(self):
        """Test that opposite vectors give pi angle."""
        r_ij = jnp.array([1.0, 0.0, 0.0])
        r_kj = jnp.array([-1.0, 0.0, 0.0])

        angle = compute_angle(r_ij, r_kj)
        assert jnp.isclose(angle, jnp.pi, atol=1e-10)

    def test_zero_angle(self):
        """Test that parallel vectors give zero angle."""
        r_ij = jnp.array([1.0, 0.0, 0.0])
        r_kj = jnp.array([2.0, 0.0, 0.0])

        angle = compute_angle(r_ij, r_kj)
        assert jnp.isclose(angle, 0.0, atol=1e-10)

    def test_arbitrary_angle(self):
        """Test an arbitrary angle (60 degrees)."""
        r_ij = jnp.array([1.0, 0.0, 0.0])
        r_kj = jnp.array([0.5, jnp.sqrt(3) / 2, 0.0])  # 60 degrees from r_ij

        angle = compute_angle(r_ij, r_kj)
        assert jnp.isclose(angle, jnp.pi / 3, atol=1e-10)


class TestTripletAnglePotential:
    """Unit tests for the triplet_angle potential function."""

    def test_angle_at_equilibrium(self):
        """Test angle potential at theta = theta0 (should be zero)."""
        k = 25.0
        theta0 = jnp.pi / 2  # 90 degrees

        # Create three particles at right angle
        centers = jnp.array([
            [1.0, 0.0, 0.0],  # i
            [0.0, 0.0, 0.0],  # j (central)
            [0.0, 1.0, 0.0],  # k
        ])
        triplet = jnp.array([0, 1, 2])
        displacement_fn, _ = space.free()

        energy = triplet_angle(centers, triplet, k, theta0, displacement_fn, USE_G96)
        assert jnp.isclose(energy, 0.0, atol=1e-10)

    def test_angle_energy_symmetric(self):
        """Test that angle energy is symmetric around equilibrium."""
        k = 25.0
        theta0 = jnp.pi / 2  # 90 degrees
        displacement_fn, _ = space.free()

        # Angle slightly less than 90 degrees
        angle_minus = jnp.pi / 2 - 0.1
        centers_minus = jnp.array([
            [jnp.cos(angle_minus), jnp.sin(angle_minus), 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        # Angle slightly more than 90 degrees
        angle_plus = jnp.pi / 2 + 0.1
        centers_plus = jnp.array([
            [jnp.cos(angle_plus), jnp.sin(angle_plus), 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        triplet = jnp.array([0, 1, 2])
        energy_minus = triplet_angle(centers_minus, triplet, k, theta0, displacement_fn, USE_G96)
        energy_plus = triplet_angle(centers_plus, triplet, k, theta0, displacement_fn, USE_G96)

        assert jnp.isclose(energy_minus, energy_plus, atol=1e-10)

    def test_angle_energy_increases_with_deviation(self):
        """Test that energy increases as angle deviates from equilibrium."""
        k = 25.0
        theta0 = jnp.pi / 2
        displacement_fn, _ = space.free()
        triplet = jnp.array([0, 1, 2])

        # At equilibrium
        centers_eq = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        # Small deviation
        centers_small = jnp.array([
            [1.0, 0.1, 0.0],  # Slight deviation
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        # Larger deviation
        centers_large = jnp.array([
            [1.0, 0.5, 0.0],  # Larger deviation
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        energy_eq = triplet_angle(centers_eq, triplet, k, theta0, displacement_fn, USE_G96)
        energy_small = triplet_angle(centers_small, triplet, k, theta0, displacement_fn, USE_G96)
        energy_large = triplet_angle(centers_large, triplet, k, theta0, displacement_fn, USE_G96)

        assert energy_eq < energy_small < energy_large
