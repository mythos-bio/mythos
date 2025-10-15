import functools
import importlib

import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

import jax_dna.energy.dna1 as dna1_energy
from jax_dna.input.topology import from_oxdna_file
from jax_dna.input.trajectory import from_file
from jax_dna.observables.base import get_duplex_quartets
from jax_dna.observables.persistence_length import PersistenceLength
from jax_dna.simulators.io import SimulatorTrajectory

TEST_INPUT = importlib.resources.files("jax_dna") / "../data/test-data/simple-helix-60bp"
TEST_N_NUC = 120
TEST_DISP_FN = jax_md.space.periodic(200)[0]

@pytest.fixture
def sim_traj():

    top = from_oxdna_file(TEST_INPUT / "sys.top")
    test_traj = from_file(
        path=TEST_INPUT / "output.dat",
        strand_lengths=top.strand_counts,
        is_oxdna=False,
    )

    return SimulatorTrajectory(rigid_body=test_traj.state_rigid_body)


@pytest.fixture
def lp_fun():
    _, config = dna1_energy.default_configs()
    geometry = config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    quartets = get_duplex_quartets(TEST_N_NUC // 2)
    return PersistenceLength(
        rigid_body_transform_fn=transform_fn, quartets=quartets, displacement_fn=TEST_DISP_FN, truncate=40
    )


def test_persistence_length_fit(lp_fun, sim_traj):
    lp = lp_fun(sim_traj)
    assert jnp.isclose(lp, 54.175785)


def test_persistence_length_fit_truncate(lp_fun, sim_traj):
    lp = lp_fun.replace(truncate=10)(sim_traj)
    assert jnp.isclose(lp, 32.105633)


def test_persistence_length_fit_weights(lp_fun, sim_traj):
    baseline = lp_fun(sim_traj)
    # For the default case, this should match what is provided with no weights
    weights = jnp.ones(sim_traj.length()) / sim_traj.length()
    lp = lp_fun(sim_traj, weights=weights)
    assert jnp.isclose(lp, baseline)

    # check with only the first frame weighted
    first_frame_baseline = lp_fun(sim_traj.slice(slice(0, 1)))
    weights = jnp.array([1] + [0]*(sim_traj.length() - 1))
    lp = lp_fun(sim_traj, weights=weights)
    assert jnp.isclose(lp, first_frame_baseline)


def test_persistence_length_fit_weights_bad_shape(lp_fun, sim_traj):
    with pytest.raises(TypeError):
        lp_fun(sim_traj, weights=np.array([1, 2]))  # wrong shape

