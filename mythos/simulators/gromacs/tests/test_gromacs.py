"""Tests for GROMACS simulator."""

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from mythos.simulators.gromacs.gromacs import GromacsSimulator
from mythos.simulators.io import SimulatorTrajectory

# Test data directory
TEST_DATA_DIR = Path("data/test-data/gromacs/sim-test-basic")


@pytest.fixture
def mock_energy_fn():
    """Create a mock energy function for testing."""

    class MockEnergyFn:
        def __call__(self, n):
            return n.sum()

        def params_dict(self, **_kwargs):
            return {}

    return MockEnergyFn()


@pytest.fixture
def gromacs_input_dir(tmp_path: Path) -> Path:
    """Create a temporary GROMACS input directory with required files."""
    # Create the required input files
    mdp_content = """; Test MDP file
integrator = md
dt = 0.002
nsteps = 100
gen-vel = yes
gen-seed = 12345
"""
    (tmp_path / "md.mdp").write_text(mdp_content)
    (tmp_path / "topol.top").write_text("; Test topology\n")
    (tmp_path / "membrane.gro").write_text("; Test structure\n")
    (tmp_path / "index.ndx").write_text("; Test index\n")

    return tmp_path


class TestGromacsSimulatorInstantiation:
    """Tests for GromacsSimulator instantiation."""

    def test_init_with_valid_dir(self, gromacs_input_dir: Path, mock_energy_fn) -> None:
        """Test that GromacsSimulator initializes with a valid directory."""
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
        )
        assert Path(sim.input_dir) == gromacs_input_dir

    def test_init_raises_for_missing_dir(self, tmp_path: Path, mock_energy_fn) -> None:
        """Test that initialization raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            GromacsSimulator(
                input_dir=tmp_path / "nonexistent",
                energy_fn=mock_energy_fn,
                binary_path="gmx",
            )

    def test_init_raises_for_missing_files(self, tmp_path: Path, mock_energy_fn):
        """Test that initialization raises FileNotFoundError when required files are missing."""
        with pytest.raises(FileNotFoundError, match="Required input file .* not found"):
            GromacsSimulator(
                input_dir=tmp_path,
                energy_fn=mock_energy_fn,
                binary_path="gmx",
            )


class TestGromacsSimulatorRun:
    """Tests for GromacsSimulator.run() method."""

    @pytest.fixture
    def mock_subprocess_and_copy_outputs(self, gromacs_input_dir: Path):
        """Create a fixture that mocks subprocess and copies test output files."""

        def _mock_check_call(cmd, cwd=None, **kwargs):
            """Mock subprocess.check_call to copy test data files."""
            # If this is the mdrun command, copy the test output files
            if cmd and "mdrun" in cmd:
                cwd_path = Path(cwd) if cwd else gromacs_input_dir
                shutil.copy(TEST_DATA_DIR / "output.tpr", cwd_path / "output.tpr")
                shutil.copy(TEST_DATA_DIR / "output.trr", cwd_path / "output.trr")
            return 0

        return _mock_check_call

    def test_run_produces_trajectory(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
        mock_subprocess_and_copy_outputs,
    ) -> None:
        """Test that run() produces a trajectory with expected shape."""
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
            binary_path="gmx",  # Mocked, so path doesn't matter
        )

        with patch("subprocess.check_call", side_effect=mock_subprocess_and_copy_outputs):
            result = sim.run(seed=42)

        # Verify the result structure
        assert result is not None
        assert hasattr(result, "observables")
        assert len(result.observables) == 1

        trajectory = result.observables[0]
        assert hasattr(trajectory, "rigid_body")
        assert hasattr(trajectory.rigid_body, "center")
        assert hasattr(trajectory.rigid_body, "orientation")

        # Check that trajectory has expected dimensions (frames, atoms, 3)
        assert len(trajectory.rigid_body.center.shape) == 3
        assert trajectory.rigid_body.center.shape[-1] == 3  # 3D positions

    def test_run_with_overwrite_false(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
        mock_subprocess_and_copy_outputs,
    ) -> None:
        """Test that run() with overwrite=False creates a copy of the input dir."""
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
            overwrite_input=False,
            binary_path="gmx",
        )

        with patch("subprocess.check_call", side_effect=mock_subprocess_and_copy_outputs):
            result = sim.run(seed=42)

        assert result is not None
        # The original directory should be unchanged (no output files)
        assert not (gromacs_input_dir / "output.trr").exists()

    def test_run_with_input_overrides(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
        mock_subprocess_and_copy_outputs,
    ) -> None:
        """Test that input overrides are applied during run."""
        overrides = {"nsteps": 500}

        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
            input_overrides=overrides,
            binary_path="gmx",
        )

        mdp_content = {
            "before": (gromacs_input_dir / "md.mdp").read_text(),
            "after": "",
        }

        def mock_subproc_store_mdp(cmd, cwd=None, **kwargs):
            mock_subprocess_and_copy_outputs(cmd, cwd=cwd, **kwargs)
            mdp_content["after"] = (Path(cwd) / "md.mdp").read_text()
            return 0

        with patch("subprocess.check_call", side_effect=mock_subproc_store_mdp):
            sim.run(seed=42)

        # With overwrite_input=True (default), the MDP should be modified in place
        assert "nsteps = 500" not in mdp_content["before"]
        assert "nsteps = 500" in mdp_content["after"]

    def test_run_sets_random_seed(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
        mock_subprocess_and_copy_outputs,
    ) -> None:
        """Test that the random seed is written to the MDP file."""
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
            binary_path="gmx",
        )

        test_seed = 98765

        mdp_content = {
            "before": (gromacs_input_dir / "md.mdp").read_text(),
            "after": "",
        }

        def mock_subproc_store_mdp(cmd, cwd=None, **kwargs):
            mock_subprocess_and_copy_outputs(cmd, cwd=cwd, **kwargs)
            mdp_content["after"] = (Path(cwd) / "md.mdp").read_text()
            return 0

        with patch("subprocess.check_call", side_effect=mock_subproc_store_mdp):
            sim.run(seed=test_seed)

        assert f"gen-seed = {test_seed}" not in mdp_content["before"]
        assert f"gen-seed = {test_seed}" in mdp_content["after"]

    def test_run_no_binary_raises(self, gromacs_input_dir: Path, mock_energy_fn) -> None:
        """Test that run() raises FileNotFoundError when GROMACS binary is missing."""
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
        )

        # do not pass the binary_path, and mock shutil.which to ensure we don't
        # find a real install!
        with patch("shutil.which", return_value=None), pytest.raises(FileNotFoundError, match="binary not found"):
            sim.run()


class TestGromacsSimulatorTrajectory:
    """Tests for trajectory reading functionality."""

    def test_trajectory_shape(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
    ) -> None:
        """Test that trajectory has correct shape from test data."""
        # Copy test output files directly to check trajectory reading
        shutil.copy(TEST_DATA_DIR / "output.tpr", gromacs_input_dir / "output.tpr")
        shutil.copy(TEST_DATA_DIR / "output.trr", gromacs_input_dir / "output.trr")

        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
        )

        trajectory = sim._read_trajectory(gromacs_input_dir)

        assert isinstance(trajectory, SimulatorTrajectory)
        # Check center positions have shape (n_frames, n_atoms, 3)
        center = trajectory.rigid_body.center
        assert len(center.shape) == 3
        assert center.shape[-1] == 3

        # Check orientations have shape (n_frames, n_atoms, 4) for quaternions
        orientation = trajectory.rigid_body.orientation.vec
        assert len(orientation.shape) == 3
        assert orientation.shape[-1] == 4

        # Verify frames and atoms match
        assert center.shape[0] == orientation.shape[0]  # Same number of frames
        assert center.shape[1] == orientation.shape[1]  # Same number of atoms
