"""Tests for Martini 2 bond potential energy function."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import MDAnalysis
import pytest
from jax_md import space
from mythos.energy.martini.m2.bond import Bond, BondConfiguration, pair_bond
from mythos.simulators.gromacs.utils import read_trajectory_mdanalysis
from mythos.simulators.io import SimulatorTrajectory

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - common jax practice


# Test data directory with GROMACS trajectory files
TEST_DATA_DIR = Path("data/test-data/martini/energy/m2/bond")


def load_bond_params() -> dict:
    """Load bond parameters from the JSON configuration file."""
    with (TEST_DATA_DIR / "bond_params.json").open() as f:
        return json.load(f)


@pytest.fixture
def gromacs_trajectory() -> SimulatorTrajectory:
    """Load the test trajectory from GROMACS output files."""
    return read_trajectory_mdanalysis(
        TEST_DATA_DIR / "test.tpr",
        TEST_DATA_DIR / "test.trr",
    )


@pytest.fixture
def bond_config() -> BondConfiguration:
    """Create a BondConfiguration from the JSON parameters file."""
    params = load_bond_params()
    return BondConfiguration(
        bond_names=tuple(params["bond_names"]),
        k=jnp.array(params["k"]),
        r0=jnp.array(params["r0"]),
    )


@pytest.fixture
def energies() -> jnp.ndarray:
    """Load reference bond energies from GROMACS output file."""
    with (TEST_DATA_DIR / "bond.xvg").open() as f:
        energy_values = []
        for line in f:
            if not line.startswith(("#", "@")):
                _, energy = line.strip().split()
                energy_values.append(float(energy))
    return jnp.array(energy_values[1:])


class TestBondConfiguration:
    """Tests for BondConfiguration."""

    def test_valid_configuration(self, bond_config: BondConfiguration):
        """Test that valid configuration is created successfully from JSON."""
        params = load_bond_params()
        n_bonds = len(params["bond_names"])

        assert len(bond_config.bond_names) == n_bonds
        assert bond_config.k.shape == (n_bonds,)
        assert bond_config.r0.shape == (n_bonds,)

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        bond_names = ("BOND1", "BOND2")
        k = jnp.array([1000.0])  # Wrong length
        r0 = jnp.array([0.47, 0.5])

        with pytest.raises(ValueError, match="must have the same length"):
            BondConfiguration(
                bond_names=bond_names,
                k=k,
                r0=r0,
            )


class TestBondEnergy:
    """Tests for Bond energy computation."""

    def test_bond_energy_against_gromacs(
        self,
        gromacs_trajectory: SimulatorTrajectory,
        bond_config: BondConfiguration,
        energies: jnp.ndarray,
    ):
        """Test bond energy calculation matches GROMACS reference values."""
        u = MDAnalysis.Universe(TEST_DATA_DIR / "test.tpr")

        bond_names = tuple(
            f"{u.atoms[b[0]].resname}_{u.atoms[b[0]].name}-{u.atoms[b[1]].name}"
            for b in u.bonds.indices
        )

        bond_fn = Bond(
            params=bond_config,
            atom_types=tuple(u.atoms.types),
            bond_names=bond_names,
            angle_names=(),
            bonded_neighbors=jnp.array(u.bonds.indices),
            unbonded_neighbors=jnp.array([]),
        )

        computed_energies = bond_fn.map(gromacs_trajectory)
        assert energies.shape[0] == gromacs_trajectory.length()
        assert jnp.allclose(computed_energies, energies)


class TestPairBondPotential:
    """Unit tests for the pair_bond potential function."""

    def test_bond_at_equilibrium(self):
        """Test bond potential at r = r0 (should be zero)."""
        k = 1250.0
        r0 = 0.47
        centers = jnp.array([[0.0, 0.0, 0.0], [r0, 0.0, 0.0]])
        pair = jnp.array([0, 1])
        displacement_fn, _ = space.free()

        energy = pair_bond(centers, pair, k, r0, displacement_fn)
        assert jnp.isclose(energy, 0.0, atol=1e-10)

    def test_bond_stretched(self):
        """Test bond potential when stretched beyond equilibrium."""
        k = 1250.0
        r0 = 0.47
        r = 0.5  # Stretched
        centers = jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        pair = jnp.array([0, 1])
        displacement_fn, _ = space.free()

        energy = pair_bond(centers, pair, k, r0, displacement_fn)
        expected = 0.5 * k * (r - r0) ** 2
        assert jnp.isclose(energy, expected)

    def test_bond_compressed(self):
        """Test bond potential when compressed below equilibrium."""
        k = 1250.0
        r0 = 0.47
        r = 0.4  # Compressed
        centers = jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        pair = jnp.array([0, 1])
        displacement_fn, _ = space.free()

        energy = pair_bond(centers, pair, k, r0, displacement_fn)
        expected = 0.5 * k * (r - r0) ** 2
        assert jnp.isclose(energy, expected)
        # Energy should be positive
        assert energy > 0.0

    def test_bond_energy_symmetric(self):
        """Test that bond energy is symmetric (compression vs stretch same displacement)."""
        k = 1250.0
        r0 = 0.47
        delta = 0.05
        displacement_fn, _ = space.free()

        # Stretched
        centers_stretched = jnp.array([[0.0, 0.0, 0.0], [r0 + delta, 0.0, 0.0]])
        energy_stretched = pair_bond(centers_stretched, jnp.array([0, 1]), k, r0, displacement_fn)

        # Compressed
        centers_compressed = jnp.array([[0.0, 0.0, 0.0], [r0 - delta, 0.0, 0.0]])
        energy_compressed = pair_bond(centers_compressed, jnp.array([0, 1]), k, r0, displacement_fn)

        assert jnp.isclose(energy_stretched, energy_compressed)
