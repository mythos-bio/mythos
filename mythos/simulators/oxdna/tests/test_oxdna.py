"""Tests for oxDNA simulator."""

import dataclasses
import importlib
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import mythos.utils.types as typ
import pytest
from mythos.input import oxdna_input
from mythos.simulators import oxdna
from mythos.simulators.base import SimulatorOutput
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
            f.write("backend = CPU\ntrajectory_file = test.conf\ntopology = test.top\nT=300K\n")

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
    with pytest.raises(FileNotFoundError, match="Input file not found at"):
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


def test_oxdna_binary_mode_ignore_params(tmp_path, mock_energy_fn, monkeypatch):
    """Test that the oxDNA simulator ignores params when configured."""
    setup_test_dir(tmp_path, add_input=True)
    monkeypatch.setattr(oxdna.oxDNASimulator, "_read_trajectory", MagicMock())
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        binary_path=shutil.which("echo"),
        ignore_params=True,
    )
    sim.run(opt_params=[{"some_param": 1.0}])

def test_oxdna_override_input(tmp_path, mock_energy_fn, monkeypatch):
    """Test that the oxDNA simulator ignores params when configured."""
    setup_test_dir(tmp_path, add_input=True)
    monkeypatch.setattr(oxdna.oxDNASimulator, "_read_trajectory", MagicMock())
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        binary_path=shutil.which("echo"),
        overwrite_input=True,
    )
    sim.run()
    assert tmp_path.joinpath("oxdna.out.log").exists()


def test_oxdna_override_keyvals(tmp_path, mock_energy_fn, monkeypatch):
    """Test that the oxDNA simulator overrides input values."""
    setup_test_dir(tmp_path, add_input=True)
    monkeypatch.setattr(oxdna.oxDNASimulator, "_read_trajectory", MagicMock())
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        binary_path=shutil.which("echo"),
        input_overrides={"steps": 10000, "T": "275K"},
        overwrite_input=True,
    )
    sim.run()
    input_content = oxdna_input.read(tmp_path / "input")
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


def test_oxdna_run(mock_energy_fn, monkeypatch):
    """Test the oxDNA simulator run function."""
    test_dir = setup_test_dir()
    monkeypatch.setattr(oxdna.oxDNASimulator, "_read_trajectory", MagicMock())
    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=mock_energy_fn,
        binary_path=shutil.which("echo"),
    )
    sim.run()
    tear_down_test_dir(test_dir)


@pytest.mark.parametrize("opt_params", [None, {"delta_backbone": 1.0}])
def test_oxdna_run_and_build_from_source(monkeypatch, tmp_path, opt_params) -> None:
    """Test for oxdna run from source with and without params."""
    setup_test_dir(tmp_path, add_input=True)
    # mock for the simulation function to "write" a trajectory file
    monkeypatch.setattr(subprocess, "check_call", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(oxdna.oxDNASimulator, "_read_trajectory", MagicMock())
    @dataclass
    class MockEF:
        params: dict = field(default_factory=dict)
        def with_params(self, new_params):
            new_params = {**self.params, **new_params}
            return dataclasses.replace(self, params=new_params)
        def params_dict(self, **_kwargs):
            return self.params

    fake_src = tmp_path.joinpath("_source/src")
    fake_src.mkdir(parents=True, exist_ok=True)
    fake_src.joinpath("model.h").write_text("#define FENE_EPS 1.0\n#define FENE_DELTA 3.0\n")

    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=MockEF(params={"eps_backbone": 0.5}),
        source_path=fake_src.parent,
    )
    traj = sim.run(opt_params=opt_params)
    assert isinstance(traj, SimulatorOutput)


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
        input_dir=tmp_path,
        new_params={
                "delta_backbone": 5.0,
                "theta0_hb_8": 1.5707963267948966,
                "a_coax_1_f6": 40.0,
                "r0_backbone": 0.756,
            },
    )
    new_lines = (tmp_path / "oxdna-build" / "model.h").read_text().splitlines()
    expected_lines = expected_model_h.read_text().splitlines()
    assert new_lines[10:] == expected_lines[10:], "model.h content does not match expected"

    original_lines = model_h.read_text().splitlines()
    assert new_lines[10:] != original_lines[10:], "model.h content was not modified"

    with pytest.raises(ValueError, match="No valid"):
        sim.build(
            input_dir=tmp_path,
            new_params={
                "a": 1,
                "b": 2,
            },
        )


def test_oxdna_simulator_trajectory_read(monkeypatch, tmp_path) -> None:
    """Test for oxdna trajectory reading after run."""

    test_dir = importlib.resources.files("mythos").parent / "data" / "test-data" / "simple-helix"

    # mock for the simulation function to "write" a trajectory file
    shutil.copytree(test_dir, tmp_path, dirs_exist_ok=True)
    def copy_traj():
        shutil.copyfile(
            test_dir / "output.dat",
            tmp_path / "output.dat",
        )
    monkeypatch.setattr(subprocess, "check_call", lambda *_args, **_kwargs: copy_traj())

    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        overwrite_input=True,  # We use this obj as shell to read, so no write
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_fn=1,
        binary_path="echo",
    )
    traj = sim._read_trajectory(tmp_path)
    assert isinstance(traj, SimulatorTrajectory)
    assert traj.rigid_body.center.shape == (100, 16, 3)

    output = sim.run()
    assert isinstance(output, SimulatorOutput)
    assert len(output.observables) == 1
    assert isinstance(output.observables[0], SimulatorTrajectory)


if __name__ == "__main__":
    test_oxdna_build()
