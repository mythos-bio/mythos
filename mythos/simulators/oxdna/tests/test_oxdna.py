"""Tests for oxDNA simulator."""

import importlib
import os
import shutil
import uuid
from pathlib import Path

import mythos.utils.types as typ
import pytest
from mythos.input import oxdna_input
from mythos.simulators import oxdna
from mythos.simulators.io import SimulatorTrajectory

file_dir = Path(os.path.realpath(__file__)).parent

@pytest.fixture
def mock_energy_fn():
    class MockEF:
        def __call__(self, n):
            return n.sum()
        def params_dict(self, **_kwargs):
            return {}
    return MockEF()


def test_guess_binary_location() -> None:
    """tests the guess_binary_location function."""

    assert oxdna._guess_binary_location("bash", "OXDNA_BIN_PATH") is not None
    with pytest.raises(FileNotFoundError):
        oxdna._guess_binary_location("zamboomafoo", "MAKE_BIN_PATH")


def setup_test_dir(test_dir: Path | None = None, add_input: bool = True):  # noqa: FBT001,FBT002
    """Setup the test directory."""
    if not test_dir:
        test_dir = file_dir / f"test_data/{uuid.uuid4()}"
        test_dir.mkdir(parents=True)
    if add_input:
        with (test_dir / "input").open("w") as f:
            f.write("trajectory_file = test.conf\ntopology = test.top\nT=300K\n")

        shutil.copyfile(
            "data/test-data/dna1/simple-helix/generated.top",
            test_dir / "test.top",
        )
        shutil.copyfile(
            "data/test-data/dna1/simple-helix/start.conf",
            test_dir / "test.conf",
        )
    return test_dir


def tear_down_test_dir(test_dir: str):
    """Tear down the test directory."""
    shutil.rmtree(test_dir)

    if len(os.listdir(Path(test_dir).parent)) == 0:
        test_dir.parent.rmdir()


def test_oxdna_init(mock_energy_fn):
    """Test the oxDNA simulator initialization."""
    test_dir = setup_test_dir()
    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        source_path="src",
    )
    tear_down_test_dir(test_dir)
    assert str(sim["input_dir"]) == str(test_dir)


def test_oxdna_run_raises_fnf(mock_energy_fn):
    """Test that the oxDNA simulator raises FileNotFoundError."""
    test_dir = setup_test_dir(add_input=False)
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        oxdna.oxDNASimulator(
            input_dir=test_dir,
            sim_type=typ.oxDNASimulatorType.DNA1,
            energy_fn=mock_energy_fn,
            source_path="src",
        )
    tear_down_test_dir(test_dir)


def test_oxdna_binary_mode_run_raises_for_missing_bin(tmp_path, mock_energy_fn):
    """Test that the oxDNA simulator raises FileNotFoundError."""
    setup_test_dir(tmp_path, add_input=True)
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        binary_path="some/non/existent/path",
    )
    with pytest.raises(FileNotFoundError, match="some/non/existent/path"):
        sim.run()


def test_oxdna_binary_mode_raises_for_params_input(tmp_path, mock_energy_fn):
    """Test that the oxDNA simulator raises ValueError."""
    setup_test_dir(tmp_path, add_input=True)
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        binary_path=shutil.which("echo"),
    )
    with pytest.raises(ValueError, match="params provided without source_path"):
        sim.run(opt_params=[{"some_param": 1.0}])


def test_oxdna_binary_mode_ignore_params(tmp_path, mock_energy_fn):
    """Test that the oxDNA simulator ignores params when configured."""
    setup_test_dir(tmp_path, add_input=True)
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        binary_path=shutil.which("echo"),
        ignore_params=True,
    )
    sim._read_trajectory = lambda: None
    sim.run(opt_params=[{"some_param": 1.0}])

def test_oxdna_override_input(tmp_path, mock_energy_fn):
    """Test that the oxDNA simulator ignores params when configured."""
    setup_test_dir(tmp_path, add_input=True)
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        binary_path=shutil.which("echo"),
        overwrite_input=True,
    )
    sim._read_trajectory = lambda: None
    sim.run()
    assert sim.base_dir == tmp_path
    assert sim.input_file == tmp_path / "input"
    assert tmp_path.joinpath("oxdna.out.log").exists()


def test_oxdna_override_keyvals(tmp_path, mock_energy_fn):
    """Test that the oxDNA simulator ignores params when configured."""
    setup_test_dir(tmp_path, add_input=True)
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        binary_path=shutil.which("echo"),
        input_overrides={"steps": 10000, "T": "275K"},
    )
    sim._read_trajectory = lambda: None
    sim.run()
    input_content = oxdna_input.read(sim.base_dir / "input")
    assert input_content["steps"] == 10000
    assert input_content["T"] == "275K"


@pytest.mark.parametrize(("bin_path", "source_path"), [(None, None), ("x", "x")])
def test_oxdna_run_raises_on_non_exclusive_bin_source_paths(bin_path, source_path):
    """Test that the oxDNA simulator raises ValueError."""
    test_dir = setup_test_dir()
    with pytest.raises(ValueError, match="Must set one and only one"):
        oxdna.oxDNASimulator(
            input_dir=test_dir,
            sim_type=typ.oxDNASimulatorType.DNA1,
            energy_fn=None,
            source_path=source_path,
            binary_path=bin_path,
        )
    tear_down_test_dir(test_dir)


def test_oxdna_run(mock_energy_fn):
    """Test the oxDNA simulator run function."""
    test_dir = setup_test_dir()
    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        binary_path=shutil.which("echo"),
    )
    sim._read_trajectory = lambda: None
    sim.run()
    with (sim.base_dir / "oxdna.out.log").open() as f:
        assert f.read() == "input\n"
    tear_down_test_dir(test_dir)


def test_oxdna_build(monkeypatch, tmp_path) -> None:
    """Test for oxdna build, fails for missing build dir"""

    model_h = Path(__file__).parent / "test_data/test.model.h"
    expected_model_h = Path(__file__).parent / "test_data/expected.model.h"

    tmp_src_dir = tmp_path / "src"
    tmp_src_dir.mkdir(parents=True, exist_ok=True)
    (tmp_src_dir / "model.h").write_text(model_h.read_text())
    tmp_path.joinpath("input").write_text("backend = CPU\n")


    monkeypatch.setenv(oxdna.CMAKE_BIN_ENV_VAR, "echo")
    monkeypatch.setenv(oxdna.MAKE_BIN_ENV_VAR, "echo")

    class MockEnergyFunction:
        def __init__(self, params):
            self.params = params

        def params_dict(self, **kwargs) -> dict:  # noqa: ARG002
            return self.params

        def with_params(self, new_params):
            return MockEnergyFunction(new_params)


    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=MockEnergyFunction({}),
        source_path=tmp_path,
    )

    sim.build(
        new_params={
                "delta_backbone": 5.0,
                "theta0_hb_8": 1.5707963267948966,
                "a_coax_1_f6": 40.0,
                "r0_backbone": 0.756,
            },
    )
    assert sim.build_dir.is_dir()
    new_lines = (sim.build_dir / "model.h").read_text().splitlines()
    expected_lines = expected_model_h.read_text().splitlines()
    assert new_lines[10:] == expected_lines[10:], "model.h content does not match expected"

    original_lines = model_h.read_text().splitlines()
    assert new_lines[10:] != original_lines[10:], "model.h content was not modified"

    with pytest.raises(ValueError, match="No valid"):
        sim.build(
            new_params={
                "a": 1,
                "b": 2,
            },
        )

    sim.cleanup_build()
    assert not sim.build_dir.exists()

def test_oxdna_simulator_trajectory_read(monkeypatch) -> None:
    """Test for oxdna trajectory reading after run."""

    test_dir = importlib.resources.files("mythos").parent / "data" / "test-data" / "simple-helix"
    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        overwrite_input=True,  # We use this obj as shell to read, so no write
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=1,
        binary_path="dummy",
    )
    traj = sim._read_trajectory()
    assert isinstance(traj, SimulatorTrajectory)
    assert traj.rigid_body.center.shape == (100, 16, 3)


if __name__ == "__main__":
    test_oxdna_build()
