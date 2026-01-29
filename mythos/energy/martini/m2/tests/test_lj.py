"""Tests for Martini 2 Lennard-Jones potential energy function."""

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import MDAnalysis
import pytest
from mythos.energy.martini.m2.lj import LJ, LJConfiguration, lennard_jones
from mythos.simulators.gromacs.utils import read_trajectory_mdanalysis
from mythos.simulators.io import SimulatorTrajectory

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - common jax practice


# Test data directory with GROMACS trajectory files
TEST_DATA_DIR = Path("data/test-data/martini/energy/m2/lj")


@dataclass
class MockSequence:
    """Mock sequence object for testing LJ energy function."""

    atom_types: tuple[str, ...]


def load_lj_params() -> dict:
    """Load LJ parameters from the JSON configuration file."""
    with (TEST_DATA_DIR / "ljconf.json").open() as f:
        return json.load(f)


def get_unbonded_neighbors(n: int, bonded_neighbors: jnp.ndarray) -> jnp.ndarray:
    """Compute unbonded neighbor pairs from bonded neighbors.

    Takes a set of bonded neighbors and returns the set of unbonded neighbors
    for a given `n` by enumerating all pairs and removing bonded neighbors.
    """
    unbonded = set(itertools.combinations(range(n), 2))
    unbonded -= {tuple(i) for i in bonded_neighbors.tolist()}
    unbonded -= {tuple(reversed(i)) for i in bonded_neighbors.tolist()}
    return jnp.array(list(unbonded))


@pytest.fixture
def gromacs_trajectory() -> SimulatorTrajectory:
    """Load the test trajectory from GROMACS output files."""
    return read_trajectory_mdanalysis(
        TEST_DATA_DIR / "test.tpr",
        TEST_DATA_DIR / "test.trr",
    )


@pytest.fixture
def lj_config() -> LJConfiguration:
    """Create a LJConfiguration from the JSON parameters file."""
    params = load_lj_params()
    # unpack into dict for config
    return LJConfiguration(
        **params
    )

@pytest.fixture
def energies() -> jnp.ndarray:
    """Load reference LJ energies from GROMACS output file."""
    with (TEST_DATA_DIR / "lj.xvg").open() as f:
        energy_values = []
        for line in f:
            if not line.startswith(("#", "@")):
                _, energy = line.strip().split()
                energy_values.append(float(energy))
    return jnp.array(energy_values[1:])


class TestLJConfiguration:
    """Tests for LJConfiguration."""

    def test_valid_configuration(self, lj_config: LJConfiguration):
        """Test that valid configuration is created successfully from JSON."""
        n_types = 37  # Known number of bead types in test config

        assert len(lj_config.bead_types) == n_types
        assert lj_config.sigmas.shape == (n_types, n_types)
        assert lj_config.epsilons.shape == (n_types, n_types)

    def test_invalid_parameter_raises_error(self):
        invalid_params = {
            "invalid_param_1": 0.5,
            "lj_sigma_A-B": 0.47,
            "lj_epsilon_A-B": 5.0,
        }
        with pytest.raises(ValueError, match="Unexpected parameter"):
            LJConfiguration(**invalid_params)

    def test_sigma_epsilon_matrix_construction(self):
        params = {
            "lj_sigma_A-A": 0.47,
            "lj_sigma_A-B": 0.50,
            "lj_sigma_B-B": 0.52,
            "lj_epsilon_A-A": 5.0,
            "lj_epsilon_A-B": 4.5,
            "lj_epsilon_B-B": 4.0,
        }
        config = LJConfiguration(**params)

        expected_bead_types = ("A", "B")
        expected_sigmas = jnp.array([[0.47, 0.50], [0.50, 0.52]])
        expected_epsilons = jnp.array([[5.0, 4.5], [4.5, 4.0]])

        assert config.bead_types == expected_bead_types
        assert jnp.allclose(config.sigmas, expected_sigmas)
        assert jnp.allclose(config.epsilons, expected_epsilons)

    @pytest.mark.parametrize(
        "incomplete_params",
        [
            {"lj_sigma_A-A": 0.47},  # Missing epsilon
            {"lj_epsilon_A-A": 5.0},  # Missing sigma
            {"lj_sigma_A-B": 0.5, "lj_epsilon_A-B": 4.5},  # Missing A-A and B-B
            {"lj_sigma_A-B": 0.5, "lj_sigma_A-A": 0.47, "lj_epsilon_A-B": 4.5},  # partial missing
        ]
    )
    def test_missing_bead_type_pairing_raises_error(self, incomplete_params):
        with pytest.raises(ValueError, match="Missing LJ"):
            LJConfiguration(**incomplete_params)

    def test_coupling_sets_multiple_params(self):
        """Test that a coupled parameter sets all underlying parameters."""
        couplings = {
            "lj_epsilon_hydrophobic": ["lj_epsilon_A-A", "lj_epsilon_B-B"],
        }
        params = {
            "lj_sigma_A-A": 0.47,
            "lj_sigma_A-B": 0.50,
            "lj_sigma_B-B": 0.52,
            "lj_epsilon_hydrophobic": 5.0,  # Coupled param
            "lj_epsilon_A-B": 4.5,
        }
        config = LJConfiguration(couplings=couplings, **params)

        # Both A-A and B-B epsilon should be set to the coupled value
        assert config.params["lj_epsilon_A-A"] == 5.0
        assert config.params["lj_epsilon_B-B"] == 5.0
        assert config.params["lj_epsilon_A-B"] == 4.5

    def test_coupling_opt_params_returns_coupled_name(self):
        """Test that opt_params returns coupled parameter name instead of individuals."""
        couplings = {
            "lj_epsilon_hydrophobic": ["lj_epsilon_A-A", "lj_epsilon_B-B"],
        }
        params = {
            "lj_sigma_A-A": 0.47,
            "lj_sigma_A-B": 0.50,
            "lj_sigma_B-B": 0.52,
            "lj_epsilon_hydrophobic": 5.0,
            "lj_epsilon_A-B": 4.5,
        }
        config = LJConfiguration(couplings=couplings, **params)

        opt = config.opt_params
        # Should have the coupled name, not the individual names
        assert "lj_epsilon_hydrophobic" in opt
        assert "lj_epsilon_A-A" not in opt
        assert "lj_epsilon_B-B" not in opt
        # Non-coupled params should still appear
        assert "lj_epsilon_A-B" in opt

    def test_coupling_getitem_access(self):
        """Test that coupled parameters can be accessed via __getitem__."""
        couplings = {
            "lj_sigma_all": ["lj_sigma_A-A", "lj_sigma_A-B", "lj_sigma_B-B"],
        }
        params = {
            "lj_sigma_all": 0.5,
            "lj_epsilon_A-A": 5.0,
            "lj_epsilon_A-B": 4.5,
            "lj_epsilon_B-B": 4.0,
        }
        config = LJConfiguration(couplings=couplings, **params)

        # Access via coupled name should work
        assert config["lj_sigma_all"] == 0.5
        # Access via individual names should also work
        assert config["lj_sigma_A-A"] == 0.5
        assert config["lj_sigma_A-B"] == 0.5

    def test_coupling_contains(self):
        """Test that __contains__ works for both coupled and individual param names."""
        couplings = {
            "lj_epsilon_coupled": ["lj_epsilon_A-A", "lj_epsilon_B-B"],
        }
        params = {
            "lj_sigma_A-A": 0.47,
            "lj_sigma_A-B": 0.50,
            "lj_sigma_B-B": 0.52,
            "lj_epsilon_coupled": 5.0,
            "lj_epsilon_A-B": 4.5,
        }
        config = LJConfiguration(couplings=couplings, **params)

        assert "lj_epsilon_coupled" in config
        assert "lj_epsilon_A-A" in config
        assert "lj_epsilon_B-B" in config
        assert "nonexistent_param" not in config

    def test_multiple_couplings(self):
        """Test that multiple independent couplings work correctly."""
        couplings = {
            "lj_sigma_diagonal": ["lj_sigma_A-A", "lj_sigma_B-B"],
            "lj_epsilon_diagonal": ["lj_epsilon_A-A", "lj_epsilon_B-B"],
        }
        params = {
            "lj_sigma_diagonal": 0.5,
            "lj_sigma_A-B": 0.48,
            "lj_epsilon_diagonal": 5.0,
            "lj_epsilon_A-B": 4.0,
        }
        config = LJConfiguration(couplings=couplings, **params)

        assert config.params["lj_sigma_A-A"] == 0.5
        assert config.params["lj_sigma_B-B"] == 0.5
        assert config.params["lj_sigma_A-B"] == 0.48
        assert config.params["lj_epsilon_A-A"] == 5.0
        assert config.params["lj_epsilon_B-B"] == 5.0
        assert config.params["lj_epsilon_A-B"] == 4.0

class TestLJEnergy:
    """Tests for LJ energy computation."""

    def test_lj_energy_against_gromacs(
        self,
        gromacs_trajectory: SimulatorTrajectory,
        lj_config: LJConfiguration,
        energies: jnp.ndarray,
    ):
        """Test LJ energy calculation matches GROMACS reference values."""
        u = MDAnalysis.Universe(TEST_DATA_DIR / "test.tpr")

        lj_fn = LJ(
            params=lj_config,
            atom_types=tuple(u.atoms.types),
            bond_names=(),
            angle_names=(),
            bonded_neighbors=jnp.array(u.bonds.indices),
            unbonded_neighbors=get_unbonded_neighbors(len(u.atoms), u.bonds.indices),
        )

        computed_energies = lj_fn.map(gromacs_trajectory)
        assert energies.shape[0] == gromacs_trajectory.length()
        assert jnp.allclose(computed_energies, energies)


class TestLennardJonesPotential:
    """Unit tests for the Lennard-Jones potential function."""

    def test_lj_at_sigma(self):
        """Test LJ potential at r = sigma (should be zero for standard LJ)."""
        sigma = 0.47
        eps = 5.0
        r = sigma

        # At r = sigma, standard LJ is zero, but shifted LJ will differ
        energy = lennard_jones(r, eps, sigma)
        assert jnp.isfinite(energy)

    def test_lj_beyond_cutoff(self):
        """Test LJ potential beyond cutoff distance (should be zero)."""
        sigma = 0.47
        eps = 5.0
        r = 2.0  # Beyond cutoff of 1.1

        energy = lennard_jones(r, eps, sigma)
        assert energy == 0.0

    def test_lj_repulsive_region(self):
        """Test LJ potential in repulsive region (r < sigma)."""
        sigma = 0.47
        eps = 5.0
        r = 0.3  # Less than sigma

        energy = lennard_jones(r, eps, sigma)
        # Energy should be positive (repulsive)
        assert energy > 0.0

    def test_lj_attractive_region(self):
        """Test LJ potential in attractive region (sigma < r < cutoff)."""
        sigma = 0.47
        eps = 5.0
        r = 0.6  # Between sigma and cutoff

        energy = lennard_jones(r, eps, sigma)
        # With shift, energy behavior depends on position relative to cutoff
        assert jnp.isfinite(energy)
