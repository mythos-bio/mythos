
import jax
import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

import jax_dna.energy.dna2 as jd_energy
import jax_dna.input.topology as jd_top
import jax_dna.input.trajectory as jd_traj
from jax_dna.utils.units import get_kt

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - ignore boolean positional value
# this is a common jax practice


COLUMN_NAMES = [
    "t",
    "fene",
    "bonded_excluded_volume",
    "stacking",
    "unbonded_excluded_volume",
    "hydrogen_bonding",
    "cross_stacking",
    "coaxial_stacking",
    "debye",
]


def get_energy_terms(base_dir: str, term: str) -> np.ndarray:
    energy_terms = np.loadtxt(base_dir + "/split_energy.dat", skiprows=1)
    return energy_terms[:, COLUMN_NAMES.index(term)]


def get_potential_energy(base_dir: str) -> np.ndarray:
    # Columns are: time, potential_energy, kinetic_energy, total_energy
    energies = np.loadtxt(base_dir + "/energy.dat")
    potential_energies = energies[:, 1]
    return potential_energies[1:]  # ignore the initial state

def get_topology(base_dir: str) -> jd_top.Topology:
    return jd_top.from_oxdna_file(base_dir + "/generated.top")


def get_trajectory(base_dir: str, topology: jd_top.Topology) -> jd_traj.Trajectory:
    return jd_traj.from_file(
        base_dir + "/output.dat",
        topology.strand_counts,
        is_oxdna=False,
    )


def get_setup_data(base_dir: str):
    topology = get_topology(base_dir)
    trajectory = get_trajectory(base_dir, topology)
    default_params = jd_energy.default_configs()[1]
    transform_fn = jd_energy.default_transform_fn()

    displacement_fn, _ = jax_md.space.periodic(20.0)

    return (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    )


@pytest.mark.parametrize("base_dir", ["data/test-data/dna2/simple-helix"])
def test_fene(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "fene")
    # compute energy terms
    energy_config = jd_energy.FeneConfiguration(**default_params["fene"])
    energy_fn = jd_energy.Fene(
        displacement_fn=displacement_fn,
        transform_fn=transform_fn,
        topology=topology,
        params=energy_config.init_params()
    )

    states = trajectory.state_rigid_body
    energy = energy_fn.map(states)
    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-6)


@pytest.mark.parametrize("base_dir", ["data/test-data/dna2/simple-helix"])
def test_bonded_excluded_volume(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "bonded_excluded_volume")
    # compute energy terms
    energy_config = jd_energy.BondedExcludedVolumeConfiguration(**default_params["bonded_excluded_volume"])
    energy_fn = jd_energy.BondedExcludedVolume(
        displacement_fn=displacement_fn,
        transform_fn=transform_fn,
        topology=topology,
        params=energy_config.init_params()
    )

    states = trajectory.state_rigid_body
    energy = energy_fn.map(states)
    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-6)


@pytest.mark.parametrize(("base_dir", "t_kelvin"), [("data/test-data/dna2/simple-helix", 296.15)])
def test_stacking(base_dir: str, t_kelvin: float):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "stacking")
    # compute energy terms
    energy_config = jd_energy.StackingConfiguration(**(default_params["stacking"] | {"kt": t_kelvin * 0.1 / 300.0}))
    energy_fn = jd_energy.Stacking(
        displacement_fn=displacement_fn,
        transform_fn=transform_fn,
        topology=topology,
        params=energy_config.init_params()
    )

    states = trajectory.state_rigid_body
    energy = energy_fn.map(states)
    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-6)


@pytest.mark.parametrize("base_dir", ["data/test-data/dna2/simple-helix"])
def test_cross_stacking(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "cross_stacking")

    default_params = jax.tree_util.tree_map(lambda arr: jnp.array(arr, dtype=jnp.float64), default_params)
    energy_config = jd_energy.CrossStackingConfiguration(**default_params["cross_stacking"])
    energy_fn = jd_energy.CrossStacking(
        displacement_fn=displacement_fn,
        transform_fn=transform_fn,
        topology=topology,
        params=energy_config.init_params()
    )

    states = trajectory.state_rigid_body
    energy = energy_fn.map(states)
    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-3)


@pytest.mark.parametrize("base_dir", ["data/test-data/dna2/simple-helix"])
def test_unbonded_excluded_volume(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "unbonded_excluded_volume")
    # compute energy terms
    energy_config = jd_energy.UnbondedExcludedVolumeConfiguration(**default_params["unbonded_excluded_volume"])
    energy_fn = jd_energy.UnbondedExcludedVolume(
        displacement_fn=displacement_fn,
        transform_fn=transform_fn,
        topology=topology,
        params=energy_config.init_params()
    )

    states = trajectory.state_rigid_body
    energy = energy_fn.map(states)
    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-6)


@pytest.mark.parametrize("base_dir", ["data/test-data/dna2/simple-helix"])
def test_hydrogen_bonding(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "hydrogen_bonding")
    # compute energy terms
    energy_config = jd_energy.HydrogenBondingConfiguration(**default_params["hydrogen_bonding"])
    energy_fn = jd_energy.HydrogenBonding(
        displacement_fn=displacement_fn,
        transform_fn=transform_fn,
        topology=topology,
        params=energy_config.init_params()
    )

    states = trajectory.state_rigid_body
    energy = energy_fn.map(states)
    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-3)


@pytest.mark.parametrize("base_dir", ["data/test-data/dna2/simple-helix", "data/test-data/dna2/simple-coax"])
def test_coaxial_stacking(base_dir: str):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "coaxial_stacking")
    # compute energy terms
    energy_config = jd_energy.CoaxialStackingConfiguration(**default_params["coaxial_stacking"])
    energy_fn = jd_energy.CoaxialStacking(
        displacement_fn=displacement_fn,
        transform_fn=transform_fn,
        topology=topology,
        params=energy_config.init_params()
    )

    states = trajectory.state_rigid_body
    energy = energy_fn.map(states)
    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-6)


@pytest.mark.parametrize(
    (
        "base_dir",
        "t_kelvin",
        "salt_conc",
        "half_charged_ends",
    ),
    [
        ("data/test-data/dna2/simple-helix", 296.15, 0.5, False),
        ("data/test-data/dna2/simple-helix-half-charged-ends", 296.15, 0.5, True),
    ],
)
def test_debye(base_dir: str, t_kelvin: float, salt_conc: float, *, half_charged_ends: bool):
    (
        topology,
        trajectory,
        default_params,
        transform_fn,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_energy_terms(base_dir, "debye")
    # compute energy terms
    kt = t_kelvin * 0.1 / 300.0
    energy_config = jd_energy.DebyeConfiguration(
        **(
            default_params["debye"]
            | {"kt": kt, "salt_conc": salt_conc, "half_charged_ends": half_charged_ends}
        )
    )
    energy_fn = jd_energy.Debye(
        displacement_fn=displacement_fn,
        transform_fn=transform_fn,
        topology=topology,
        params=energy_config.init_params()
    )

    states = trajectory.state_rigid_body
    energy = energy_fn.map(states)
    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-3)


@pytest.mark.parametrize(
        "base_dir",
        ["data/test-data/dna2/simple-helix", "data/test-data/dna2/simple-helix-half-charged-ends"]
)
def test_debye_is_end_initialization(base_dir: str):
    top, _, _, transform_fn, _ = get_setup_data(base_dir)
    dummy = 1 # just to satisfy interface, but make clear here it is not used
    debye_from_top = jd_energy.Debye(
        transform_fn=transform_fn,
        displacement_fn=dummy,
        topology=top,
        params=dummy,
    )
    assert (debye_from_top.is_end == top.is_end).all()
    debye_from_direct = jd_energy.Debye(
        transform_fn=transform_fn,
        displacement_fn=dummy,
        bonded_neighbors=top.bonded_neighbors,
        unbonded_neighbors=top.unbonded_neighbors,
        is_end=top.is_end,
        seq=dummy,
        params=dummy,
    )
    assert (debye_from_direct.is_end == top.is_end).all()
    with pytest.raises(ValueError, match="is_end must be provided"):
        jd_energy.Debye(
            transform_fn=transform_fn,
            displacement_fn=dummy,
            bonded_neighbors=top.bonded_neighbors,
            unbonded_neighbors=top.unbonded_neighbors,
            seq=dummy,
            params=dummy,
        )


@pytest.mark.parametrize(
    ("base_dir", "half_charged_ends"),
    [
        ("data/test-data/dna2/simple-helix", False),
        ("data/test-data/dna2/simple-helix-half-charged-ends", True),
    ]
)
def test_total_energy(base_dir: str, *, half_charged_ends: bool):
    (
        topology,
        trajectory,
        _,
        _,
        displacement_fn,
    ) = get_setup_data(base_dir)

    terms = get_potential_energy(base_dir)

    energy_fn = jd_energy.create_default_energy_fn(
        topology=topology,
        displacement_fn=displacement_fn,
    ).with_params(
        kt = get_kt(296.15),
        half_charged_ends = half_charged_ends,
    )

    states = trajectory.state_rigid_body
    energy = energy_fn.map(states)
    energy = np.around(energy / topology.n_nucleotides, 6)
    np.testing.assert_allclose(energy, terms, atol=1e-3)


if __name__ == "__main__":
    pytest.main()
