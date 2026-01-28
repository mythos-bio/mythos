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
    return LJConfiguration(
        bead_types=tuple(params["bead_types"]),
        sigmas=jnp.array(params["sigmas"]),
        epsilons=jnp.array(params["epsilons"]),
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
        params = load_lj_params()
        n_types = len(params["bead_types"])

        assert len(lj_config.bead_types) == n_types
        assert lj_config.sigmas.shape == (n_types, n_types)
        assert lj_config.epsilons.shape == (n_types, n_types)

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise ValueError."""
        bead_types = ("P1", "P2")
        sigmas = jnp.ones((2, 2))
        epsilons = jnp.ones((3, 3))  # Wrong shape

        with pytest.raises(ValueError, match="must have shape"):
            LJConfiguration(
                bead_types=bead_types,
                sigmas=sigmas,
                epsilons=epsilons,
            )


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
