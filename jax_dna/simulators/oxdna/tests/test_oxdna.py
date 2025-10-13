"""Tests for oxDNA simulator."""

import os
import shutil
import uuid
from pathlib import Path

import jax_dna.utils.types as typ
import pytest
from jax_dna.simulators import oxdna

file_dir = Path(os.path.realpath(__file__)).parent


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
            f.write("trajectory_file = test.conf\ntopology = test.top\n")

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


def test_oxdna_init():
    """Test the oxDNA simulator initialization."""
    test_dir = setup_test_dir()
    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
        source_path="src",
    )
    tear_down_test_dir(test_dir)
    assert str(sim["input_dir"]) == str(test_dir)


def test_oxdna_run_raises_fnf():
    """Test that the oxDNA simulator raises FileNotFoundError."""
    test_dir = setup_test_dir(add_input=False)
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        oxdna.oxDNASimulator(
            input_dir=test_dir,
            sim_type=typ.oxDNASimulatorType.DNA1,
            energy_configs=[],
            source_path="src",
        )
    tear_down_test_dir(test_dir)


def test_oxdna_binary_mode_run_raises_for_missing_bin(tmp_path):
    """Test that the oxDNA simulator raises FileNotFoundError."""
    setup_test_dir(tmp_path, add_input=True)
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
        binary_path="some/non/existent/path",
    )
    with pytest.raises(FileNotFoundError, match="some/non/existent/path"):
        sim.run()


def test_oxdna_binary_mode_raises_for_params_input(tmp_path):
    """Test that the oxDNA simulator raises ValueError."""
    setup_test_dir(tmp_path, add_input=True)
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
        binary_path=shutil.which("echo"),
    )
    with pytest.raises(ValueError, match="params provided without source_path"):
        sim.run(opt_params=[{"some_param": 1.0}])


def test_oxdna_binary_mode_ignore_params(tmp_path):
    """Test that the oxDNA simulator ignores params when configured."""
    setup_test_dir(tmp_path, add_input=True)
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
        binary_path=shutil.which("echo"),
        ignore_params=True,
    )
    sim._read_trajectory = lambda: None
    sim.run(opt_params=[{"some_param": 1.0}])

def test_oxdna_override_input(tmp_path):
    """Test that the oxDNA simulator ignores params when configured."""
    setup_test_dir(tmp_path, add_input=True)
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
        binary_path=shutil.which("echo"),
        overwrite_input=True,
    )
    sim._read_trajectory = lambda: None
    sim.run()
    assert sim.base_dir == tmp_path
    assert sim.input_file == tmp_path / "input"
    assert tmp_path.joinpath("oxdna.out.log").exists()


@pytest.mark.parametrize(("bin_path", "source_path"), [(None, None), ("x", "x")])
def test_oxdna_run_raises_on_non_exclusive_bin_source_paths(bin_path, source_path):
    """Test that the oxDNA simulator raises ValueError."""
    test_dir = setup_test_dir()
    with pytest.raises(ValueError, match="Must set one and only one"):
        oxdna.oxDNASimulator(
            input_dir=test_dir,
            sim_type=typ.oxDNASimulatorType.DNA1,
            energy_configs=[],
            source_path=source_path,
            binary_path=bin_path,
        )
    tear_down_test_dir(test_dir)


def test_oxdna_run():
    """Test the oxDNA simulator run function."""
    test_dir = setup_test_dir()
    sim = oxdna.oxDNASimulator(
        input_dir=test_dir,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[],
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
    excepted_model_h = Path(__file__).parent / "test_data/expected.model.h"

    tmp_src_dir = tmp_path / "src"
    tmp_src_dir.mkdir(parents=True, exist_ok=True)
    (tmp_src_dir / "model.h").write_text(model_h.read_text())
    tmp_path.joinpath("input").write_text("backend = CPU\n")


    monkeypatch.setenv(oxdna.CMAKE_BIN_ENV_VAR, "echo")
    monkeypatch.setenv(oxdna.MAKE_BIN_ENV_VAR, "echo")

    class MockEnergyConfig:
        def __init__(self, params):
            self.params = params

        def init_params(self) -> "MockEnergyConfig":
            return self

        def to_dictionary(self, include_dependent, exclude_non_optimizable) -> dict:  # noqa: ARG002
            return self.params

        def __or__(self, other: dict):
            return MockEnergyConfig(self.params | other)

    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
        sim_type=typ.oxDNASimulatorType.DNA1,
        energy_configs=[MockEnergyConfig({}), MockEnergyConfig({})],
        source_path=tmp_path,
    )

    sim.build(
        new_params=[
            {
                "FENE_DELTA": 5.0,
                "HYDR_THETA8_T0": 1.5707963267948966,
                "HYDR_T3_MESH_POINTS": "HYDR_T2_MESH_POINTS",
                "CXST_T5_MESH_POINTS": 6,
            },
            {},
        ]
    )
    assert sim.build_dir.is_dir()
    assert (sim.build_dir / "model.h").read_text().splitlines()[-10:] != excepted_model_h.read_text().splitlines()[-10:]

    sim.cleanup_build()
    assert not sim.build_dir.exists()


if __name__ == "__main__":
    test_oxdna_build()
