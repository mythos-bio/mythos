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

import numpy as np
import pandas as pd
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
            energy_fn=mock_energy_fn,
            source_path="src",
        )
    tear_down_test_dir(test_dir)


def test_oxdna_binary_mode_run_raises_for_missing_bin(tmp_path, mock_energy_fn):
    """Test that the oxDNA simulator raises FileNotFoundError."""
    setup_test_dir(tmp_path, add_input=True)
    sim = oxdna.oxDNASimulator(
        input_dir=tmp_path,
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
        energy_fn=1,
        binary_path="echo",
    )
    traj = sim._read_trajectory(tmp_path)
    assert isinstance(traj, SimulatorTrajectory)
    assert traj.center.shape == (100, 16, 3)

    output = sim.run()
    assert isinstance(output, SimulatorOutput)
    assert len(output.observables) == 1
    assert isinstance(output.observables[0], SimulatorTrajectory)


def setup_umbrella_test_dir(
        test_dir: Path, umbrella_sampling: int = 1, *, include_keys: bool = True, num_order_params: int = 1
    ) -> None:
    """Setup a test directory configured for umbrella sampling."""
    test_dir.mkdir(parents=True, exist_ok=True)

    input_content = "backend = CPU\ntrajectory_file = test.conf\ntopology = test.top\nT=300K\n"
    if include_keys:
        input_content += f"umbrella_sampling = {umbrella_sampling}\n"
        input_content += "order_parameters = op.txt\n"
        input_content += "weights_file = wfile.txt\n"
        input_content += "energy_file = energy.dat\n"
        input_content += "last_hist_file = last_hist.dat\n"
        input_content += "op_file = op.txt\n"

    with (test_dir / "input").open("w") as f:
        f.write(input_content)

    # Create order parameters file
    with (test_dir / "op.txt").open("w") as f:
        for i in range(num_order_params):
            f.write(f"{{\norder_parameter = op_{i}\nname = op_name_{i}\n}}\n\n")

    # Create weights file
    with (test_dir / "wfile.txt").open("w") as f:
        for line in range(5):
            op_prefix = " ".join(str(line) for i in range(num_order_params))
            f.write(f"{op_prefix} 1.0\n")

    shutil.copyfile(
        "data/test-data/dna1/simple-helix/generated.top",
        test_dir / "test.top",
    )
    shutil.copyfile(
        "data/test-data/dna1/simple-helix/start.conf",
        test_dir / "test.conf",
    )
    return test_dir


class TestOxDNAUmbrellaSampler:

    @pytest.mark.parametrize("keys", [
        ("umbrella_sampling", "order_parameters"),
        ("weights_file", "order_parameters"),
        ("umbrella_sampling", "weights_file"),
    ])
    def test_umbrella_sampler_raises_for_missing_umbrella_sampling_key(self, tmp_path, mock_energy_fn, keys):
        setup_umbrella_test_dir(tmp_path, include_keys=False)
        # Add only some keys, missing umbrella_sampling
        with (tmp_path / "input").open("a") as f:
            for k in keys:
                f.write(f"{k} = test_value\n")

        with pytest.raises(ValueError, match="Missing required umbrella sampling config"):
            oxdna.oxDNAUmbrellaSampler(
                input_dir=tmp_path,
                energy_fn=mock_energy_fn,
                source_path="src",
            )

    def test_umbrella_sampler_raises_for_wrong_umbrella_sampling_value(self, tmp_path, mock_energy_fn):
        """Test that ValueError is raised when umbrella_sampling is not 1."""
        setup_umbrella_test_dir(tmp_path, umbrella_sampling=0)

        with pytest.raises(ValueError, match="umbrella_sampling must be set to 1"):
            oxdna.oxDNAUmbrellaSampler(
                input_dir=tmp_path,
                energy_fn=mock_energy_fn,
                source_path="src",
            )

    def test_umbrella_sampler_init_success(self, tmp_path, mock_energy_fn):
        setup_umbrella_test_dir(tmp_path, umbrella_sampling=1)

        sim = oxdna.oxDNAUmbrellaSampler(
            input_dir=tmp_path,
            energy_fn=mock_energy_fn,
            source_path="src",
        )
        assert str(sim["input_dir"]) == str(tmp_path)

    @pytest.mark.parametrize("num_order_params", [1, 2])
    def test_umbrella_sampler_run_produces_energy_and_weights(
        self, tmp_path, mock_energy_fn, monkeypatch, num_order_params
    ):
        setup_umbrella_test_dir(tmp_path, umbrella_sampling=1, num_order_params=num_order_params)

        # Create mock energy data file (umbrella sampling format)
        # Columns: time, potential_energy, acc_ratio_trans, acc_ratio_rot,
        # acc_ratio_vol, order_param, weight
        op = " ".join("0" for _ in range(num_order_params))
        energy_data = f"0 -10.5 0.5 0.5 0.0 {op} 1.0\n1 -11.0 0.5 0.5 0.0 {op} 1.0\n2 -10.8 0.5 0.5 0.0 {op} 1.0\n"
        with (tmp_path / "energy.dat").open("w") as f:
            f.write("# header line to skip\n")
            f.write(energy_data)

        # Create mock last histogram file
        # Columns: order_param, count, unbiased_count
        hist_data = f"{op} 10 5\n{op} 8 4\n{op} 12 6\n"
        with (tmp_path / "last_hist.dat").open("w") as f:
            f.write("# header line to skip\n")
            f.write(hist_data)

        # Mock the parent run_simulation to return a valid output
        mock_trajectory = MagicMock(spec=SimulatorTrajectory)
        mock_output = SimulatorOutput(observables=[mock_trajectory], state={})

        monkeypatch.setattr(
            oxdna.oxDNASimulator, "run_simulation",
            lambda self, input_dir, **kwargs: mock_output,  # noqa: ARG005
        )

        sim = oxdna.oxDNAUmbrellaSampler(
            input_dir=tmp_path,
            energy_fn=mock_energy_fn,
            source_path="src",
        )

        output = sim.run_simulation(tmp_path)

        # Check that output has both trajectory and energy observables
        assert len(output.observables) == 2
        assert output.observables[0] is mock_trajectory
        assert isinstance(output.observables[1], oxdna.UmbrellaEnergyInfo)

        # Check that the energy dataframe has expected columns
        energy_df = output.observables[1]
        assert "time" in energy_df.columns
        assert "potential_energy" in energy_df.columns
        assert "weight" in energy_df.columns

        # Check that weights are computed in state
        assert "weights" in output.state
        weights = output.state["weights"]
        assert isinstance(weights, pd.DataFrame)
        # Weights should be recomputed based on inverse unbiased_count
        # of 5,4,6
        assert np.allclose(weights["weights"], [1.2, 1.5, 1.0])
        # We should have the right number of columns in output. Since we write
        # the index, for this test we reset it to get the shape.
        assert weights.reset_index().shape[1] == 1 + num_order_params

    def test_umbrella_sampler_writes_weights_file(self, tmp_path, mock_energy_fn, monkeypatch):
        setup_umbrella_test_dir(tmp_path, umbrella_sampling=1)

        # Create mock energy data file
        with (tmp_path / "energy.dat").open("w") as f:
            f.write("# header\n")
            f.write("0 -10.5 0.5 0.5 0.0 0.1 1.0\n")

        # Create mock histogram file
        with (tmp_path / "last_hist.dat").open("w") as f:
            f.write("# header\n")
            f.write("0.1 10 5\n")

        mock_trajectory = MagicMock(spec=SimulatorTrajectory)
        mock_output = SimulatorOutput(observables=[mock_trajectory], state={})
        monkeypatch.setattr(
            oxdna.oxDNASimulator, "run_simulation",
            lambda self, input_dir, **kwargs: mock_output,  # noqa: ARG005
        )

        sim = oxdna.oxDNAUmbrellaSampler(
            input_dir=tmp_path,
            energy_fn=mock_energy_fn,
            source_path="src",
        )

        # Create custom weights to pass in
        custom_weights = pd.DataFrame({"idx": [0, 1], "w": [2.0, 3.0]})
        sim.run_simulation(tmp_path, weights=custom_weights)

        # Check that the weights file was written
        weights_content = (tmp_path / "wfile.txt").read_text()
        assert "2.0" in weights_content
        assert "3.0" in weights_content


if __name__ == "__main__":
    test_oxdna_build()
