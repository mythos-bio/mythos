
from unittest import mock

import chex
import numpy as np
import pytest
from jax_dna.energy.configuration import BaseConfiguration
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


@chex.dataclass(frozen=True)
class DummyConfig(BaseConfiguration):
    eps_backbone: float = 1.0
    delta_backbone: float = 2.0
    r0_backbone: float = 3.0

    def init_params(self):
        return self


@pytest.fixture
def dummy_input_lines():
    def dummy_params(num):
        return " ".join([f"{i+1}.0" for i in range(num)])
    return [
        "variable seed equal 123",
        "dump out all custom 1 trajectory.dat id mol type x y z ix iy iz vx vy vz &",
        "    c_quat[1] c_quat[2] c_quat[3] c_quat[4] angmomx angmomy angmomz",
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
def dummy_input_dir(tmp_path):
    input_file = tmp_path / "input"
    input_file.write_text(
    )
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
    with pytest.raises(ValueError, match="Dump line not found in input"):
        _lammps_oxdna_replace_inputs(input_lines, params, seed=42)


def test_lammps_oxdna_replace_inputs_missing_seed(dummy_input_lines):
    no_seed = [line for line in dummy_input_lines if "variable seed" not in line]
    params = {"eps_backbone": 1.1, "delta_backbone": 2.2, "r0_backbone": 3.3}
    with pytest.raises(ValueError, match="Random seed not specified in input"):
        _lammps_oxdna_replace_inputs(no_seed, params, seed=42)


def test_lammps_oxdna_replace_inputs_wrong_traj_name(dummy_input_lines):
    lines = [line.replace("trajectory.dat", "wrong_name.dat") for line in dummy_input_lines]
    with pytest.raises(ValueError, match="Expected dump filename"):
        _lammps_oxdna_replace_inputs(lines, {}, None)


def test_lammps_oxdna_replace_inputs_dump_missing_fields(dummy_input_lines):
    lines = [line.replace("angmomx", "") for line in dummy_input_lines]
    with pytest.raises(ValueError, match="missing required fields"):
        _lammps_oxdna_replace_inputs(lines, {}, None)


def test_lammps_oxdna_replace_errors_on_missing_pair_coeff(dummy_input_lines):
    lines = [line for line in dummy_input_lines if "pair_coeff * * oxdna/excv" not in line]
    with pytest.raises(ValueError, match="Missing oxdna pair parameters"):
        _lammps_oxdna_replace_inputs(lines, {}, seed=42)


def test_lammps_oxdna_replace_inputs_random_seed(dummy_input_lines):
    # replace with value that is not in random range
    lines = [line if "variable seed" not in line else "variable seed equal NONINT" for line in dummy_input_lines]
    params = {"eps_backbone": 1.1, "delta_backbone": 2.2, "r0_backbone": 3.3}
    out = _lammps_oxdna_replace_inputs(lines, params, seed=None)
    seed_line = next(line for line in out if "variable seed" in line)
    int(seed_line.split()[-1])  # should be convertible to int


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
    sim = LAMMPSoxDNASimulator(input_dir=input_dir, energy_configs=[DummyConfig()])
    assert sim.input_dir != input_dir
    assert sim.input_dir.joinpath("input").exists()


def test_simulator_run_mocks_subprocess(tmp_path, dummy_input_lines, dummy_trajectory_data):
    # Prepare input file
    input_file = tmp_path / "input"
    input_file.write_text("\n".join(dummy_input_lines))
    tmp_path.joinpath("trajectory.dat").write_text(dummy_trajectory_data)
    # Create a dummy trajectory.dat file to be read
    sim = LAMMPSoxDNASimulator(
        input_dir=tmp_path,
        overwrite=True,
        energy_configs=[DummyConfig()],
    )
    params = [{"eps_backbone": 1.1, "delta_backbone": 2.2, "r0_backbone": 3.3}]
    # Patch subprocess.check_call so it doesn't actually run anything
    with mock.patch("subprocess.check_call") as m_call:
        m_call.return_value = None
        result = sim.run(params, seed=123)
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
