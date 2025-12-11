import functools
import importlib

import jax
import jax.numpy as jnp
import jax_md
import pytest

import mythos.energy.dna1 as jdna1_energy
import mythos.input.topology as jdna_top
from mythos.energy.base import ComposedEnergyFunction
from mythos.input.trajectory import from_file
from mythos.observables.melting_temp import MeltingTemp, jax_interp1d
from mythos.simulators.io import SimulatorTrajectory
from mythos.simulators.oxdna.utils import read_energy
from mythos.utils.units import get_kt

jax.config.update("jax_enable_x64", val=True)


KT_RANGE = get_kt(jnp.linspace(280, 350, 20))
EXAMPLE_FINF = jnp.array([
    9.54661208e-01, 9.30589100e-01, 8.94485689e-01, 8.41267687e-01,
    7.64995709e-01, 6.60578535e-01, 5.27863030e-01, 3.77845592e-01,
    2.35185564e-01, 1.26478977e-01, 6.05169415e-02, 2.70268531e-02,
    1.17361152e-02, 5.08819979e-03, 2.24055667e-03, 1.01612059e-03,
    4.81753103e-04, 2.43235060e-04, 1.33714983e-04, 8.17874679e-05,
])
KT = 0.10238333333333333 # corresponds to temp in example data
DATA_PATH = importlib.resources.files("mythos") / ".." / "data" / "test-data" / "melting_temp"

@pytest.fixture
def energy_info():
    energy_fns = jdna1_energy.default_energy_fns()
    energy_configs = jdna1_energy.default_energy_configs()
    geometry = jdna1_energy.default_configs()[1]["geometry"]
    transform_fn = functools.partial(
        jdna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )
    top = jdna_top.from_oxdna_file(DATA_PATH / "sys.top")
    energy_fn = ComposedEnergyFunction.from_lists(
        energy_fns=energy_fns,
        energy_configs=energy_configs,
        transform_fn=transform_fn,
        displacement_fn=jax_md.space.periodic(20.0)[0],
        topology=top,
    ).with_noopt(
        "ss_stack_weights", "ss_hb_weights", "kt"
    ).with_params(kt=KT)

    return energy_fn, energy_configs, transform_fn


@pytest.fixture
def melting_temp_fn(energy_info):
    energy_fn, energy_config, transform_fn = energy_info
    melting_temp = MeltingTemp(
        rigid_body_transform_fn=transform_fn,
        sim_temperature=KT,
        temperature_range=KT_RANGE,
        energy_fn=energy_fn,
    )
    return melting_temp, energy_fn.opt_params()


@pytest.fixture
def traj_info():
    def read_traj():
        top = jdna_top.from_oxdna_file(DATA_PATH / "sys.top")
        traj = from_file(DATA_PATH / "trajectory.dat", top.strand_counts, is_5p_3p=False)
        return SimulatorTrajectory(rigid_body=traj.state_rigid_body)

    traj = read_traj()
    energy = read_energy(DATA_PATH)
    return traj, energy

def test_interp():
    assert jnp.isclose(jax_interp1d(EXAMPLE_FINF, KT_RANGE, 0.5), 0.1009298)


def test_melting_temp_calculation(melting_temp_fn, traj_info):
    melting_temp, opt_params = melting_temp_fn
    traj, energy = traj_info

    tm = melting_temp(
        trajectory=traj,
        bind_states=energy["bond"].to_numpy(),
        umbrella_weights=energy["weight"].to_numpy(),
        opt_params=opt_params
    )
    assert jnp.isclose(tm, 0.1009298)


def test_melting_curve(melting_temp_fn, traj_info):
    melting_temp, opt_params = melting_temp_fn
    traj, energy = traj_info
    temps, curve = melting_temp.get_melting_curve(
        trajectory=traj,
        bind_states=energy["bond"].to_numpy(),
        umbrella_weights=energy["weight"].to_numpy(),
        opt_params=opt_params
    )
    assert jnp.allclose(temps, KT_RANGE)
    assert jnp.allclose(curve, EXAMPLE_FINF)




