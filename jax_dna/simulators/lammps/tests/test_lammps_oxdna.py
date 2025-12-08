
import importlib
from pathlib import Path
from unittest import mock

import chex
import jax
import jax_md
import numpy as np
import pytest
from jax_dna.energy import dna2
from jax_dna.energy.dna1 import create_default_energy_fn
from jax_dna.input.topology import from_oxdna_file
from jax_dna.input.trajectory import Trajectory
from jax_dna.simulators.io import SimulatorTrajectory
from jax_dna.simulators.lammps.lammps_oxdna import (
    LAMMPS_REQUIRED_FIELDS,
    LAMMPSoxDNASimulator,
    _lammps_oxdna_replace_inputs,
    _read_lammps_output,
    _replace_parts_in_line,
    _transform_lammps_quat,
    _transform_lammps_state,
    _transform_param,
)

jax.config.update("jax_enable_x64", val=True)


@chex.dataclass(frozen=True)
class DummyFunction:
    eps_backbone: float = 1.0
    delta_backbone: float = 2.0
    r0_backbone: float = 3.0

    def params_dict(self, **_kwargs) -> dict[str, float]:
        return {
            "eps_backbone": self.eps_backbone,
            "delta_backbone": self.delta_backbone,
            "r0_backbone": self.r0_backbone,
        }

    def with_params(self, params: dict[str, float]) -> "DummyFunction":
        return DummyFunction(
            eps_backbone=params.get("eps_backbone", self.eps_backbone),
            delta_backbone=params.get("delta_backbone", self.delta_backbone),
            r0_backbone=params.get("r0_backbone", self.r0_backbone),
        )


@pytest.fixture
def dummy_input_lines():
    def dummy_params(num):
        return " ".join([f"{i+1}.0" for i in range(num)])
    return [
        "variable myvar equal 0",
        "variable another_var equal 42",
        "variable seed equal 123",
        "dump unusable all custom 1 unusable.dat id x y z",
        "dump out all custom 1 trajectory.dat id mol type x y z ix iy iz vx vy vz &",
        "    c_quat[1] c_quat[2] c_quat[3] c_quat[4] angmomx angmomy angmomz",
        "dump_modify out sort id",
        "bond_coeff * 1.0 2.0 3.0",
        "pair_coeff * * oxdna/excv " + dummy_params(9),
        "pair_coeff * * oxdna/stk " + dummy_params(22),
        "pair_coeff * * oxdna/hbond " + dummy_params(25),
        "pair_coeff 1 4 oxdna/hbond " + dummy_params(25),
        "pair_coeff 2 3 oxdna/hbond " + dummy_params(25),
        "pair_coeff * * oxdna/xstk " + dummy_params(23),
        "pair_coeff * * oxdna/coaxstk" + dummy_params(21),
    ]


@pytest.fixture
def dummy_input_dir(tmp_path, dummy_input_lines):
    input_file = tmp_path / "input"
    input_file.write_text("\n".join(dummy_input_lines))
    return tmp_path


@pytest.fixture
def dummy_input_lines_dna2():
    def dummy_params(num):
        return " ".join([f"{i+1}.0" for i in range(num)])
    return [
        "variable seed equal 123",
        "dump unusable all custom 1 unusable.dat id x y z",
        "dump out all custom 1 trajectory.dat id mol type x y z ix iy iz vx vy vz &",
        "    c_quat[1] c_quat[2] c_quat[3] c_quat[4] angmomx angmomy angmomz",
        "dump_modify out sort id",
        "bond_coeff * 1.0 2.0 3.0",
        "pair_coeff * * oxdna2/excv " + dummy_params(9),
        "pair_coeff * * oxdna2/stk " + dummy_params(22),
        "pair_coeff * * oxdna2/hbond " + dummy_params(25),
        "pair_coeff 1 4 oxdna2/hbond " + dummy_params(25),
        "pair_coeff 2 3 oxdna2/hbond " + dummy_params(25),
        "pair_coeff * * oxdna2/xstk " + dummy_params(23),
        "pair_coeff * * oxdna2/coaxstk" + dummy_params(19),
        "pair_coeff * * oxdna2/dh " + dummy_params(3),
    ]


@pytest.fixture
def dummy_input_dir_dna2(tmp_path, dummy_input_lines_dna2):
    input_file = tmp_path / "input"
    input_file.write_text("\n".join(dummy_input_lines_dna2))
    return tmp_path


@pytest.fixture
def dummy_trajectory_timestep0():
    return (
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id mol type x y z ix iy iz vx vy vz c_quat[1] c_quat[2] c_quat[3] c_quat[4] "
        "angmomx angmomy angmomz\n"
        "  should not be read...  \n"
    )


@pytest.fixture
def dummy_trajectory_timestep1():
    return (
        "ITEM: TIMESTEP\n1\n"
        "ITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id mol type x y z ix iy iz vx vy vz c_quat[1] c_quat[2] c_quat[3] c_quat[4] "
        "angmomx angmomy angmomz\n"
        "1 1 1 -6.126400967010039e-01 -6.573324218675992e-01  3.629709915928486e-01 0 0 0  "
        "7.617309940416418e-02  8.357491234620849e-03  2.079367646633826e-01  9.963616537366130e-01 "
        "-5.113495195424776e-02  8.493495051523761e-03 -6.765007164584903e-02  2.081677204968046e-01 "
        "-2.236562803957254e-01 -3.447898004843278e-01\n"
        "2 1 2 -4.529946722258457e-01 -1.000366856409590e+00  7.238570846152456e-01 0 0 0 "
        "-4.076500292276172e-02  2.687673978112082e-01 -2.167598496593139e-01  9.370916221621590e-01  "
        "1.106573889544250e-01  6.689473390339626e-02  3.242519522212261e-01 -1.618550301407265e-01 "
        "-6.754152355326958e-02  3.479753155340162e-02\n"
    )


@pytest.fixture
def dummy_trajectory_data(dummy_trajectory_timestep0, dummy_trajectory_timestep1):
    return dummy_trajectory_timestep0 + dummy_trajectory_timestep1


@pytest.fixture
def dummy_trajectory_file(tmp_path, dummy_trajectory_data):
    file = tmp_path / "trajectory.dat"
    file.write_text(dummy_trajectory_data)
    return file


def test_transform_param_neg_cos():
    assert _transform_param("neg_cos_phi1_star_stack", 2.5) == -2.5
    assert _transform_param("other_param", 2.5) == 2.5


def test_replace_parts_in_line_basic():
    inputs = "1.0 2.0 3.0"
    replacements = ("eps_backbone", "delta_backbone", "r0_backbone")
    params = {"eps_backbone": 1.1, "delta_backbone": 2.2, "r0_backbone": 3.3}
    out = _replace_parts_in_line(inputs, replacements, params)
    assert np.allclose(np.fromstring(out, sep=" "), np.array(list(params.values())))


def test_lammps_oxdna_replace_inputs_success(dummy_input_lines):
    params = {"eps_backbone": 1.1, "delta_backbone": 2.2, "r0_backbone": 3.3}
    out = _lammps_oxdna_replace_inputs(dummy_input_lines, params, seed=42)
    assert any("variable seed equal 42" in line for line in out)
    assert any("bond_coeff * 1.100000 2.200000 3.300000" in line for line in out)


def test_lammps_oxdna_replace_inputs_missing_dump():
    input_lines = [
        "variable seed equal 123",
        "bond_coeff * 1.0 2.0 3.0",
    ]
    params = {"eps_backbone": 1.1, "delta_backbone": 2.2, "r0_backbone": 3.3}
    with pytest.raises(ValueError, match="Required dump not found"):
        _lammps_oxdna_replace_inputs(input_lines, params, seed=42)


def test_lammps_oxdna_replace_inputs_missing_seed(dummy_input_lines):
    no_seed = [line for line in dummy_input_lines if "variable seed" not in line]
    params = {"eps_backbone": 1.1, "delta_backbone": 2.2, "r0_backbone": 3.3}
    with pytest.raises(ValueError, match="Missing variable for replacements: seed"):
        _lammps_oxdna_replace_inputs(no_seed, params, seed=42)


def test_lammps_oxdna_replace_inputs_wrong_traj_name(dummy_input_lines):
    lines = [line.replace("trajectory.dat", "wrong_name.dat") for line in dummy_input_lines]
    with pytest.raises(ValueError, match="Required dump not found"):
        _lammps_oxdna_replace_inputs(lines, {}, None)


def test_lammps_oxdna_replace_inputs_dump_missing_fields(dummy_input_lines):
    lines = [line.replace("angmomx", "") for line in dummy_input_lines]
    with pytest.raises(ValueError, match="Required dump not found"):
        _lammps_oxdna_replace_inputs(lines, {}, None)


def test_lammps_oxdna_replace_inputs_random_seed(dummy_input_lines):
    # replace with value that is not in random range
    lines = [line if "variable seed" not in line else "variable seed equal NONINT" for line in dummy_input_lines]
    params = {"eps_backbone": 1.1, "delta_backbone": 2.2, "r0_backbone": 3.3}
    out = _lammps_oxdna_replace_inputs(lines, params, seed=None)
    seed_line = next(line for line in out if "variable seed" in line)
    int(seed_line.split()[-1])  # should be convertible to int


def test_lammps_oxdna_replace_inputs_with_variables(dummy_input_lines):
    out = _lammps_oxdna_replace_inputs(dummy_input_lines, {}, seed=42, variables={"myvar": 10, "another_var": 99})
    myvar_line = next(line for line in out if "variable myvar" in line)
    another_var_line = next(line for line in out if "variable another_var" in line)
    assert myvar_line == "variable myvar equal 10"
    assert another_var_line == "variable another_var equal 99"


def test_lammps_oxdna_replace_inputs_with_variables_missing(dummy_input_lines):
    with pytest.raises(ValueError, match="Missing variable for replacements: nonexistentvar"):
        _lammps_oxdna_replace_inputs(dummy_input_lines, {}, seed=42, variables={"nonexistentvar": 123})


def test_transform_lammps_quat_shape_and_values():
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    out = _transform_lammps_quat(quat)
    assert out.shape == (6,)
    assert np.isclose(out[0], 1.0)


def test_transform_lammps_state_shape():
    state = np.zeros(len(LAMMPS_REQUIRED_FIELDS))
    out = _transform_lammps_state(state, list(LAMMPS_REQUIRED_FIELDS))
    assert out.shape == (3 + 6 + 3 + 3,)


def test_simulator_post_init(tmp_path, dummy_input_lines):
    input_dir = tmp_path
    (input_dir / "input").write_text("\n".join(dummy_input_lines))
    sim = LAMMPSoxDNASimulator(input_dir=input_dir, energy_fn=DummyFunction())
    assert sim.input_dir != input_dir
    assert sim.input_dir.joinpath("input").exists()

@pytest.mark.parametrize(
        ("variables_arg", "expected_lines"),
        [
            ({}, ["variable myvar equal 0", "variable another_var equal 42"]),
            (
                {"variables": {"myvar": 10, "another_var": 99}},
                ["variable myvar equal 10", "variable another_var equal 99"]
            ),
        ]
)
def test_simulator_run_mocks_subprocess(
    tmp_path,
    dummy_input_lines,
    dummy_trajectory_data,
    variables_arg,
    expected_lines,
):
    # Prepare input file
    input_file = tmp_path / "input"
    input_file.write_text("\n".join(dummy_input_lines))
    tmp_path.joinpath("trajectory.dat").write_text(dummy_trajectory_data)
    # Create a dummy trajectory.dat file to be read
    sim = LAMMPSoxDNASimulator(
        input_dir=tmp_path,
        overwrite=True,
        energy_fn=DummyFunction(),
        **variables_arg,  # use args unpacking here to ensure we test default (iso None input)
    )
    params = {"eps_backbone": 1.1, "delta_backbone": 2.2, "r0_backbone": 3.3}
    def get_fene_line(file):
        lines = file.read_text().splitlines()
        fene_line = next(line for line in lines if line.startswith("bond_coeff * "))
        return np.fromstring(fene_line.replace("bond_coeff * ", ""), sep=" ")

    # Patch subprocess.check_call so it doesn't actually run anything
    with mock.patch("subprocess.check_call") as m_call:
        m_call.return_value = None
        # sanity check - to protect against test data change that would invalidate the test
        assert not np.allclose(get_fene_line(input_file), np.array(list(params.values())))

        result = sim.run(params, seed=123)
        # Params above are updates of the fene params, ensure that we wrote them
        # out correctly in the input file.
        assert np.allclose(get_fene_line(input_file), np.array(list(params.values())))

        # Check that variables were set correctly
        all_lines = input_file.read_text().splitlines()
        assert all(line in all_lines for line in expected_lines)

        # outputs check, we've already written a dummy trajectory.dat file
        assert isinstance(result, SimulatorTrajectory)
        assert result.length() == 1
        m_call.assert_called_once()


def test_lammps_read_trajectory_missing_state_fields(tmp_path, dummy_trajectory_data):
    bad_data = dummy_trajectory_data.replace("c_quat[4]", "")
    dummy_trajectory_file = tmp_path / "trajectory.dat"
    dummy_trajectory_file.write_text(bad_data)
    with pytest.raises(ValueError, match="missing required fields"):
        _read_lammps_output(dummy_trajectory_file)


@pytest.mark.parametrize("nstates", [1, 3])
def test_read_lammps_output(tmp_path, nstates, dummy_trajectory_data, dummy_trajectory_timestep1):
    # Create a minimal trajectory.dat file
    expected_states = np.array([
        np.array([
            -0.6126401, -0.65733242,  0.36297099,  0.99070266, -0.1356765 ,
            -0.01000662, 0.02384375,  0.10074864,  0.99462615,  0.13535469,
            0.01485072,  0.36949023,  0.31555816, -0.33903703, -0.52266142
        ]),
        np.array([
            -0.45299467, -1.00036686,  0.72385708,  0.78077153,  0.62251237,
            -0.05361124,  0.19713474, -0.16401073,  0.96656007, -0.07243678,
            0.47758235,  -0.38516829, -0.24535349, -0.10238513,  0.05274903
        ]),
    ])

    file = tmp_path / "trajectory.dat"
    file.write_text(dummy_trajectory_data + dummy_trajectory_timestep1 * (nstates - 1))
    traj = _read_lammps_output(file)
    assert isinstance(traj, Trajectory)
    assert hasattr(traj, "n_nucleotides")
    assert traj.n_nucleotides == 2
    assert len(traj.times) == nstates
    assert len(traj.states) == nstates
    for i in range(nstates):
        assert isinstance(traj.states[i].array, np.ndarray)
        assert np.allclose(traj.states[i].array, expected_states)  # same for all states


# Helper function for the default energy replacement tests below
def _check_replacement_coeff_lines(lines, expected, skiplist, shouldmatch):
    lines_map = {}
    for line in lines:
        for key in expected:
            if line.startswith(key):
                lines_map[key] = line.replace(key, "").strip()

    for key, values_str in expected.items():
        replaced = lines_map[key]
        updated_values = np.fromstring(replaced, sep=" ", dtype=np.float64)
        skips = skiplist.get(key, 0)
        updated_values = updated_values[skips:]
        expected_values = np.fromstring(values_str, sep=" ", dtype=np.float64)
        isclose = np.allclose(updated_values, expected_values)
        if shouldmatch:
            assert isclose, f"Mismatch in line: {key}"
        else:
            assert not isclose, f"Expected mismatch in line: {key}"


def test_check_dna1_default_energy_fn_replacements(dummy_input_dir):
    # Expected comes from https://docs.lammps.org/pair_oxdna.html - which has
    # the default energy function parameters frozen in for oxdna model in
    # LAMMPS. The non-parameter values have been stripped, noting that the first
    # parameter from the first hbond is also not included.
    expected = {
        "pair_coeff * * oxdna/excv": "2.0 0.7 0.675 2.0 0.515 0.5 2.0 0.33 0.32",
        "pair_coeff * * oxdna/stk":
            "1.3448 2.6568 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65",
        "pair_coeff * * oxdna/hbond": (
            "8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46"
            " 3.141592653589793 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45"),
        "pair_coeff 1 4 oxdna/hbond": (
            "1.077 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793"
            " 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45"),
        "pair_coeff 2 3 oxdna/hbond": (
            "1.077 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793"
            " 0.7 4.0 1.5707963267948966 0.45 4.0 1.5707963267948966 0.45"),
        "pair_coeff * * oxdna/xstk": (
            "47.5 0.575 0.675 0.495 0.655 2.25 0.791592653589793 0.58 1.7 1.0 0.68 1.7 1.0 0.68"
            " 1.5 0 0.65 1.7 0.875 0.68 1.7 0.875 0.68"),
        "pair_coeff * * oxdna/coaxstk":
            "46.0 0.4 0.6 0.22 0.58 2.0 2.541592653589793 0.65 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 -0.65 2.0 -0.65"
    }

    skiplist = {
        "pair_coeff * * oxdna/stk": 2,
        "pair_coeff * * oxdna/hbond": 2,
        "pair_coeff 1 4 oxdna/hbond": 1,
        "pair_coeff 2 3 oxdna/hbond": 1,
    }

    # sanity check - default params should NOT match expected
    _check_replacement_coeff_lines(
        dummy_input_dir.joinpath("input").read_text().splitlines(), expected, skiplist, shouldmatch=False
    )

    energy_fn = create_default_energy_fn(topology=mock.MagicMock())
    sim = LAMMPSoxDNASimulator(input_dir=dummy_input_dir, energy_fn=energy_fn, overwrite=True)
    sim._replace_parameters(params={}, seed=42)

    _check_replacement_coeff_lines(
        sim.input_dir.joinpath("input").read_text().splitlines(), expected, skiplist, shouldmatch=True
    )

def test_check_dna2_default_energy_fn_replacements(dummy_input_dir_dna2):
    # Expected comes from https://docs.lammps.org/pair_oxdna2.html - which has
    # the default energy function parameters frozen in for oxdna model in
    # LAMMPS. The non-parameter values have been stripped, noting that the first
    # parameter from the first hbond is also not included.
    expected = {
        "pair_coeff * * oxdna2/excv": "2.0 0.7 0.675 2.0 0.515 0.5 2.0 0.33 0.32",
        "pair_coeff * * oxdna2/stk":
            "1.3523 2.6717 6.0 0.4 0.9 0.32 0.75 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 2.0 0.65 2.0 0.65",
        "pair_coeff * * oxdna2/hbond": (
            "8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 "
            "1.5707963267948966 0.45 4.0 1.5707963267948966 0.45"),
        "pair_coeff 1 4 oxdna2/hbond": (
            "1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 "
            "1.5707963267948966 0.45 4.0 1.5707963267948966 0.45"),
        "pair_coeff 2 3 oxdna2/hbond": (
            "1.0678 8.0 0.4 0.75 0.34 0.7 1.5 0 0.7 1.5 0 0.7 1.5 0 0.7 0.46 3.141592653589793 0.7 4.0 "
            "1.5707963267948966 0.45 4.0 1.5707963267948966 0.45"),
        "pair_coeff * * oxdna2/xstk": (
            "47.5 0.575 0.675 0.495 0.655 2.25 0.791592653589793 0.58 1.7 1.0 0.68 1.7 1.0 0.68 1.5 0 "
            "0.65 1.7 0.875 0.68 1.7 0.875 0.68"),
        "pair_coeff * * oxdna2/coaxstk":
            "58.5 0.4 0.6 0.22 0.58 2.0 2.891592653589793 0.65 1.3 0 0.8 0.9 0 0.95 0.9 0 0.95 40.0 3.116592653589793",
        "pair_coeff * * oxdna2/dh": "0.5 0.815",
    }

    skiplist = {
        "pair_coeff * * oxdna2/stk": 2,
        "pair_coeff * * oxdna2/hbond": 2,
        "pair_coeff 1 4 oxdna2/hbond": 1,
        "pair_coeff 2 3 oxdna2/hbond": 1,
        "pair_coeff * * oxdna2/dh": 1,
    }

    # sanity check - default params should NOT match expected
    _check_replacement_coeff_lines(
        dummy_input_dir_dna2.joinpath("input").read_text().splitlines(), expected, skiplist, shouldmatch=False
    )

    energy_fn = dna2.create_default_energy_fn(topology=mock.MagicMock())
    sim = LAMMPSoxDNASimulator(input_dir=dummy_input_dir_dna2, energy_fn=energy_fn, overwrite=True)
    sim._replace_parameters(params={}, seed=42)

    _check_replacement_coeff_lines(
        sim.input_dir.joinpath("input").read_text().splitlines(), expected, skiplist, shouldmatch=True
    )


def _read_lammps_energies(output_file: Path, read_num) -> np.ndarray:
    # helper for reading energies from lammps log file, for the specific output
    # style in this test. This is defined here in test because we don't really
    # want to support the generic functionality at the moment (if ever).
    with Path(output_file).open("r") as f:
        for line in f:
            if line.startswith("   Step         PotEng"):
                f.readline() # skip first state (this is init state which we don't use)
                energies_str = [f.readline().strip().split()[1] for _ in range(read_num)]
                return np.float64(energies_str)
        return None


def test_lammps_energy():
    # Test we compute the same energy as LAMMPS for a known system with the
    # default energy parameters for oxdna1. The simulation has been run using
    # the inputs from data/templates/simple-helix-60bp-oxdna1-lammps using
    #   lmp -in input
    # It is important to note that we must exclude the BondedExcludedVolume energy
    # since LAMMPS does not implement it.
    indir = importlib.resources.files("jax_dna") / "../data/templates/simple-helix-60bp-oxdna1-lammps"
    topology = from_oxdna_file(indir / "sys.top")
    traj = _read_lammps_output(indir / "trajectory.dat")
    sim_traj = SimulatorTrajectory(rigid_body=traj.state_rigid_body)

    # box and kT match the box from lammps init conf, and temp for lammps input
    energy_fn = create_default_energy_fn(
        topology=topology,
        displacement_fn=jax_md.space.periodic(200)[0]
    ).without_terms("BondedExcludedVolume"  # lammps doesn't do this term
    ).with_params(kt = 0.1)
    energy = energy_fn.map(sim_traj.rigid_body)

    # lammps will report per-nucleotide energy
    energy_per_nucleotide = energy / topology.n_nucleotides
    expected_energies = _read_lammps_energies(indir / "log.lammps", read_num=sim_traj.length())

    assert np.allclose(energy_per_nucleotide, expected_energies)


def test_lammps_energy_dna2():
    indir = importlib.resources.files("jax_dna") / "../data/templates/simple-helix-60bp-oxdna2-lammps"
    topology = from_oxdna_file(indir / "sys.top")
    traj = _read_lammps_output(indir / "trajectory.dat")
    sim_traj = SimulatorTrajectory(rigid_body=traj.state_rigid_body)

    # box and kT match the box from lammps init conf, and temp for lammps input
    energy_fn = dna2.create_default_energy_fn(
        topology=topology,
        displacement_fn=jax_md.space.periodic(200)[0]
    ).without_terms("BondedExcludedVolume"  # lammps doesn't do this term
    ).with_params(
        # To be explicit. half_charged_ends is False in LAMMPS, and unsure if it
        # is configurable there.
        kt = 0.1, salt_conc=0.5, q_eff=0.815, half_charged_ends=False
    )
    energy = energy_fn.map(sim_traj.rigid_body)

    # lammps will report per-nucleotide energy
    energy_per_nucleotide = energy / topology.n_nucleotides
    expected_energies = _read_lammps_energies(indir / "log.lammps", read_num=sim_traj.length())

    assert np.allclose(energy_per_nucleotide, expected_energies)
