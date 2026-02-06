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
            # We set the binary to "gmx" in tests, ensure that the command
            # the simulator wants to run is actually "gmx"
            assert cmd[0] == "gmx", "Expected 'gmx' command in subprocess call"
            # If this is the mdrun command, copy the test output files
            if cmd and "mdrun" in cmd:
                cwd_path = Path(cwd) if cwd else gromacs_input_dir
                shutil.copy(TEST_DATA_DIR / "output.tpr", cwd_path / "output.tpr")
                shutil.copy(TEST_DATA_DIR / "output.trr", cwd_path / "output.trr")
                # Create output.gro to simulate the structure output from mdrun
                (cwd_path / "output.gro").write_text("; Output structure\n")
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

        with (
            patch("subprocess.check_call", side_effect=mock_subprocess_and_copy_outputs),
            patch.object(GromacsSimulator, "_update_topology_params"),
        ):
            result = sim.run(seed=42)

        # Verify the result structure
        assert result is not None
        assert hasattr(result, "observables")
        assert len(result.observables) == 1

        trajectory = result.observables[0]
        assert hasattr(trajectory, "center")
        assert hasattr(trajectory, "orientation")

        # Check that trajectory has expected dimensions (frames, atoms, 3)
        assert trajectory.length() == 10
        assert trajectory.center.shape[-1] == 3  # 3D positions

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

        with (
            patch("subprocess.check_call", side_effect=mock_subprocess_and_copy_outputs),
            patch.object(GromacsSimulator, "_update_topology_params"),
        ):
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
            prod_mdp = Path(cwd) / "production_md.mdp"
            mdp_content["after"] = prod_mdp.read_text()
            return 0

        with (
            patch("subprocess.check_call", side_effect=mock_subproc_store_mdp),
            patch.object(GromacsSimulator, "_update_topology_params"),
        ):
            sim.run(seed=42)

        # The overrides should be written to the production MDP file
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
            prod_mdp = Path(cwd) / "production_md.mdp"
            mdp_content["after"] = prod_mdp.read_text()
            return 0

        with (
            patch("subprocess.check_call", side_effect=mock_subproc_store_mdp),
            patch.object(GromacsSimulator, "_update_topology_params"),
        ):
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

    def test_zero_equilibration_runs_production_only(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
        mock_subprocess_and_copy_outputs,
    ) -> None:
        """Test that equilibration_steps=0 runs only the production step."""
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
            equilibration_steps=0,
            binary_path="gmx",
        )

        calls = []

        def tracking_check_call(cmd, cwd=None, **kwargs):
            calls.append({"cmd": cmd, "cwd": cwd})
            return mock_subprocess_and_copy_outputs(cmd, cwd=cwd, **kwargs)

        with (
            patch("subprocess.check_call", side_effect=tracking_check_call),
            patch.object(GromacsSimulator, "_update_topology_params"),
        ):
            result = sim.run(seed=42)

        assert result is not None

        # Should have exactly 2 subprocess calls: grompp + mdrun for production only
        assert len(calls) == 2

        # First call should be grompp for production
        grompp_cmd = calls[0]["cmd"]
        assert "grompp" in grompp_cmd
        assert "production_md.mdp" in grompp_cmd

        # grompp should use the original structure file (not output.gro)
        c_flag_idx = grompp_cmd.index("-c")
        assert grompp_cmd[c_flag_idx + 1] == "membrane.gro"

        # Second call should be mdrun for production
        assert "mdrun" in calls[1]["cmd"]

        # No equilibration MDP should have been written
        assert not (gromacs_input_dir / "equilibration_md.mdp").exists()

    def test_positive_equilibration_runs_two_stages(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
        mock_subprocess_and_copy_outputs,
    ) -> None:
        """Test that equilibration_steps>0 runs equilibration then production."""
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
            equilibration_steps=500,
            binary_path="gmx",
        )

        calls = []

        def tracking_check_call(cmd, cwd=None, **kwargs):
            calls.append({"cmd": cmd, "cwd": cwd})
            return mock_subprocess_and_copy_outputs(cmd, cwd=cwd, **kwargs)

        with (
            patch("subprocess.check_call", side_effect=tracking_check_call),
            patch.object(GromacsSimulator, "_update_topology_params"),
        ):
            result = sim.run(seed=42)

        assert result is not None

        # Should have exactly 4 subprocess calls:
        # grompp (equil) + mdrun (equil) + grompp (prod) + mdrun (prod)
        assert len(calls) == 4

        # --- Equilibration stage ---
        eq_grompp_cmd = calls[0]["cmd"]
        assert "grompp" in eq_grompp_cmd
        assert "equilibration_md.mdp" in eq_grompp_cmd

        # Equilibration grompp should use the original structure file
        c_flag_idx = eq_grompp_cmd.index("-c")
        assert eq_grompp_cmd[c_flag_idx + 1] == "membrane.gro"

        eq_mdrun_cmd = calls[1]["cmd"]
        assert "mdrun" in eq_mdrun_cmd

        # --- Production stage ---
        prod_grompp_cmd = calls[2]["cmd"]
        assert "grompp" in prod_grompp_cmd
        assert "production_md.mdp" in prod_grompp_cmd

        # Production grompp should use output.gro from equilibration
        c_flag_idx = prod_grompp_cmd.index("-c")
        assert prod_grompp_cmd[c_flag_idx + 1] == "output.gro"

        prod_mdrun_cmd = calls[3]["cmd"]
        assert "mdrun" in prod_mdrun_cmd

    def test_equilibration_mdp_overrides(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
        mock_subprocess_and_copy_outputs,
    ) -> None:
        """Test that equilibration MDP has nsteps, nstxout=0, nstenergy=0."""
        eq_steps = 1000
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
            equilibration_steps=eq_steps,
            binary_path="gmx",
        )

        mdp_contents = {}

        def tracking_check_call(cmd, cwd=None, **kwargs):
            cwd_path = Path(cwd) if cwd else gromacs_input_dir
            # Capture MDP file contents when grompp is called
            if cmd and "grompp" in cmd:
                f_flag_idx = cmd.index("-f")
                mdp_name = cmd[f_flag_idx + 1]
                mdp_path = cwd_path / mdp_name
                if mdp_path.exists():
                    mdp_contents[mdp_name] = mdp_path.read_text()
            return mock_subprocess_and_copy_outputs(cmd, cwd=cwd, **kwargs)

        with (
            patch("subprocess.check_call", side_effect=tracking_check_call),
            patch.object(GromacsSimulator, "_update_topology_params"),
        ):
            sim.run(seed=42)

        # Verify equilibration MDP overrides
        eq_mdp = mdp_contents["equilibration_md.mdp"]
        assert f"nsteps = {eq_steps}" in eq_mdp
        assert "nstxout = 0" in eq_mdp
        assert "nstenergy = 0" in eq_mdp

        # Verify production MDP does NOT have the equilibration-specific overrides
        prod_mdp = mdp_contents["production_md.mdp"]
        assert "nstxout = 0" not in prod_mdp
        assert "nstenergy = 0" not in prod_mdp

    def test_equilibration_with_simulation_steps(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
        mock_subprocess_and_copy_outputs,
    ) -> None:
        """Test that simulation_steps overrides production nsteps but not equilibration."""
        eq_steps = 200
        sim_steps = 5000
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
            equilibration_steps=eq_steps,
            simulation_steps=sim_steps,
            binary_path="gmx",
        )

        mdp_contents = {}

        def tracking_check_call(cmd, cwd=None, **kwargs):
            cwd_path = Path(cwd) if cwd else gromacs_input_dir
            if cmd and "grompp" in cmd:
                f_flag_idx = cmd.index("-f")
                mdp_name = cmd[f_flag_idx + 1]
                mdp_path = cwd_path / mdp_name
                if mdp_path.exists():
                    mdp_contents[mdp_name] = mdp_path.read_text()
            return mock_subprocess_and_copy_outputs(cmd, cwd=cwd, **kwargs)

        with (
            patch("subprocess.check_call", side_effect=tracking_check_call),
            patch.object(GromacsSimulator, "_update_topology_params"),
        ):
            sim.run(seed=42)

        # Equilibration should use equilibration_steps, not simulation_steps
        eq_mdp = mdp_contents["equilibration_md.mdp"]
        assert f"nsteps = {eq_steps}" in eq_mdp
        assert f"nsteps = {sim_steps}" not in eq_mdp

        # Production should use simulation_steps
        prod_mdp = mdp_contents["production_md.mdp"]
        assert f"nsteps = {sim_steps}" in prod_mdp


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
        center = trajectory.center
        assert len(center.shape) == 3
        assert center.shape[-1] == 3

        # Check orientations have shape (n_frames, n_atoms, 4) for quaternions
        orientation = trajectory.orientation.vec
        assert len(orientation.shape) == 3
        assert orientation.shape[-1] == 4

        # Verify frames and atoms match
        assert center.shape[0] == orientation.shape[0]  # Same number of frames
        assert center.shape[1] == orientation.shape[1]  # Same number of atoms


class TestUpdateTopologyParams:
    """Tests for _update_topology_params method."""

    def test_update_topology_params_creates_preprocessed_file(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
    ) -> None:
        """Test that _update_topology_params creates the preprocessed topology file."""
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
            binary_path="gmx",
        )

        # The preprocessed topology file that should be created
        pp_topology_file = gromacs_input_dir / "_pp_topol.top"

        def mock_grompp_creates_pp_file(cmd, cwd=None, **kwargs):
            """Mock subprocess that creates the preprocessed topology file."""
            # Only create the file if this is the grompp command with -pp flag
            if cmd and "grompp" in cmd and "-pp" in cmd:
                pp_topology_file.write_text("; Preprocessed topology\n")
            return 0

        with (
            patch("subprocess.check_call", side_effect=mock_grompp_creates_pp_file),
            patch("mythos.simulators.gromacs.gromacs.replace_params_in_topology") as mock_replace,
        ):
            sim._update_topology_params({"test_param": 1.0})

        # Verify the preprocessed file was created
        assert pp_topology_file.exists()

        # Verify replace_params_in_topology was called with the right arguments
        mock_replace.assert_called_once_with(
            pp_topology_file,
            {"test_param": 1.0},
            pp_topology_file,
        )

    def test_update_topology_params_raises_when_file_not_created(
        self,
        gromacs_input_dir: Path,
        mock_energy_fn,
    ) -> None:
        """Test that _update_topology_params raises FileNotFoundError when grompp fails to create file."""
        sim = GromacsSimulator(
            input_dir=gromacs_input_dir,
            energy_fn=mock_energy_fn,
            binary_path="gmx",
        )

        def mock_grompp_no_output(cmd, cwd=None, **kwargs):
            """Mock subprocess that does NOT create the preprocessed file."""
            return 0

        with (
            patch("subprocess.check_call", side_effect=mock_grompp_no_output),
            pytest.raises(FileNotFoundError, match="Preprocessed topology file not found"),
        ):
            sim._update_topology_params({})
