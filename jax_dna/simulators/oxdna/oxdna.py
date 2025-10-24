"""OXDNA sampler module.

Run an jax_dna simulation using an oxDNA sampler.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

import chex
import numpy as np

import jax_dna.input.oxdna_input as jd_oxdna
import jax_dna.input.topology as jd_top
import jax_dna.input.trajectory as jd_traj
import jax_dna.simulators.base as jd_base
import jax_dna.simulators.io as jd_sio
import jax_dna.simulators.oxdna.utils as oxdna_utils
import jax_dna.utils.types as jd_types
from jax_dna.energy.base import EnergyFunction

ERR_OXDNA_NOT_FOUND = "OXDNA binary not found at: {}"
ERR_MISSING_REQUIRED_KEYS = "Missing required keys: {}"
ERR_INPUT_FILE_NOT_FOUND = "Input file not found: {}"
ERR_OXDNA_FAILED = "OXDNA simulation failed"
OXDNA_TRAJECTORY_FILE_KEY = "trajectory_file"
OXDNA_TOPOLOGY_FILE_KEY = "topology"

ERR_BUILD_SETUP_FAILED = "OXDNA build setup failed wiht return code: {}"
ERR_ORIG_MODEL_H_NOT_FOUND = "Original model.h file not found, looked at {}"

MAKE_BIN_ENV_VAR = "MAKE_BIN_PATH"
CMAKE_BIN_ENV_VAR = "CMAKE_BIN_PATH"

logger = logging.getLogger(__name__)


# We do not force the user the set this because they may not be recompiling oxDNA
def _guess_binary_location(bin_name: str, env_var: str) -> Path | None:
    """Guess the location of a binary."""
    if bin_loc := os.environ.get(env_var, shutil.which(bin_name)):
        return bin_loc
    raise FileNotFoundError(f"executable {bin_loc}")


@chex.dataclass
class oxDNASimulator(jd_base.BaseSimulation):  # noqa: N801 oxDNA is a special word
    """A sampler base on running an oxDNA simulation."""

    input_dir: Path
    sim_type: jd_types.oxDNASimulatorType
    energy_fn: EnergyFunction
    n_build_threads: int = 4
    logger_config: dict[str, typing.Any] | None = None
    binary_path: Path | None = None
    source_path: Path | None = None
    ignore_params: bool = False
    overwrite_input: bool = False


    def __post_init__(self, *args, **kwds) -> None:
        """Check the validity of the configuration."""
        if not (bool(self.binary_path) ^ bool(self.source_path)):
            raise ValueError("Must set one and only one of binary_path or source_path")

        self.input_dir = Path(self.input_dir).resolve()
        self.base_dir = self.input_dir
        if self.source_path or not self.overwrite_input:
            self.base_dir = Path(tempfile.mkdtemp(prefix="jaxdna-oxdna-sim-")).resolve()

        self.build_dir = None
        if self.source_path:
            self.source_path = Path(self.source_path).resolve()
            self.build_dir = self.base_dir / "oxdna-build"
            self.binary_path = self.build_dir / "bin" / "oxDNA"
        self.binary_path = Path(self.binary_path).resolve()

        if not self.overwrite_input:
            shutil.copytree(self.input_dir, self.base_dir, dirs_exist_ok=True)

        self.input_file = Path(self.base_dir) / "input"
        self.input_config = jd_oxdna.read(self.input_file)
        self._initialize_logger()

    def _initialize_logger(self) -> None:
        config = self.logger_config if self.logger_config is not None else {}
        level = config.get("level", logging.INFO)
        logger = logging.getLogger(__name__)
        logger.setLevel(level)

        if config.get("filename", None):
            handler = logging.FileHandler(config["filename"])
            handler.setLevel(level)
        else:
            handler = logging.StreamHandler()
            handler.setLevel(level)

        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
        self._logger = logger

    def use_cached_build(self, binary_path: Path) -> None:
        """Switch to use a precompiled binary.

        This may be useful when running on a cluster with a shared file system,
        or running on a single machine, particularly in cases where:
            N_simulators * n_build_threads > N_cpu_cores.

        Caution: the user is responsible for ensuring that the binary at
        provided path is pre-built for the appropriate parameter set, there is
        no check performed at simulation run-time to verify this.
        """
        self.source_path = None
        self.binary_path = binary_path
        self.ignore_params = True

    def run(
        self,
        opt_params: list[jd_types.Params] | None = None,
        seed: np.ndarray | None = None,
        **_,
    ) -> jd_traj.Trajectory:
        """Run an oxDNA simulation."""
        if opt_params is not None:
            if self.source_path:
                self.build(new_params=opt_params)
            elif not self.ignore_params:
                raise ValueError("params provided without source_path. Set ignore_params to override")
        elif self.source_path and not self.binary_path.exists():
            self.build(new_params=[])

        logger.info("oxDNA input file: %s", self.input_file)

        # overwrite the seed
        self.input_config["seed"] = seed or np.random.default_rng().integers(0, 2**32)
        jd_oxdna.write(self.input_config, self.input_file)

        # remove existing trajectory and energy files (others?), otherwise they
        # will be appended to
        for output in ["trajectory_file", "energy_file"]:
            if file := self.input_config.get(output, None):
                self.base_dir.joinpath(file).unlink(missing_ok=True)

        std_out_file = self.base_dir / "oxdna.out.log"
        std_err_file = self.base_dir / "oxdna.err.log"
        logger.info("Starting oxDNA simulation")
        logger.debug(
            "oxDNA std_out->%s, std_err->%s",
            std_out_file,
            std_err_file,
        )
        with std_out_file.open("w") as f_std, std_err_file.open("w") as f_err:
            cmd = [self.binary_path, "input"]
            logger.debug("running command: %s", cmd)
            subprocess.check_call(cmd, stdout=f_std, stderr=f_err, cwd=self.base_dir) #noqa: S603 false positive
        logger.info("oxDNA simulation complete")

        return self._read_trajectory()

    def _read_trajectory(self) -> jd_sio.SimulatorTrajectory:
        trajectory_file = self.base_dir / self.input_config["trajectory_file"]
        topology_file = self.base_dir / self.input_config["topology"]

        topology = jd_top.from_oxdna_file(topology_file)
        trajectory = jd_traj.from_file(trajectory_file, topology.strand_counts, is_oxdna=False)

        logger.debug("oxDNA trajectory com size: %s", trajectory.state_rigid_body.center.shape)

        return jd_sio.SimulatorTrajectory(
            rigid_body=trajectory.state_rigid_body,
        )


    def build(self, *, new_params: list[dict]) -> None:
        """Update the simulation.

        This function will recompile the oxDNA binary with the new parameters.
        """
        cmake_bin = _guess_binary_location("cmake", CMAKE_BIN_ENV_VAR)
        make_bin = _guess_binary_location("make", MAKE_BIN_ENV_VAR)

        logger.info("Updating oxDNA parameters (build path: %s)", str(self.build_dir))

        self.build_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("build_dir: %s", self.build_dir)

        model_h = self.build_dir / "model.h"
        if not model_h.exists():
            model_h.write_text(self.source_path.joinpath("src/model.h").read_text())

        updated_params = self.energy_fn.with_params(new_params).params_dict(exclude_non_optimizable=True)
        oxdna_utils.update_params(model_h, updated_params)

        std_out = self.build_dir / "jax_dna.cmake.std.log"
        std_err = self.build_dir / "jax_dna.cmake.err.log"

        if not (self.build_dir / "CMakeLists.txt").exists():
            with std_out.open("w") as f_std, std_err.open("w") as f_err:
                cmd = [cmake_bin, self.source_path, f"-DCMAKE_CXX_FLAGS=--include {model_h}"]
                if self.input_config["backend"] == "CUDA":
                    cmd = [*cmd, "-DCUDA=ON", "-DCUDA_COMMON_ARCH=OFF"]
                logger.debug("Attempting cmake using (std_out->%s, std_err->%s): %s", std_out, std_err, cmd)
                subprocess.check_call(cmd, shell=False, cwd=self.build_dir, stdout=f_std, stderr=f_err)  # noqa: S603 false positive

            logger.debug("cmake completed")

        # rebuild the binary
        std_out = self.build_dir / "jax_dna.make.std.log"
        std_err = self.build_dir / "jax_dna.make.err.log"
        logger.debug(
            "running make with %d processes: std_out->%s, std_err->%s",
            self.n_build_threads,
            std_out,
            std_err,
        )
        with std_out.open("w") as f_std, std_err.open("w") as f_err:
            subprocess.check_call(
                [make_bin, f"-j{self.n_build_threads}", "clean", "oxDNA"],  # clean since model.h is not tracked
                shell=False,  # noqa: S603 false positive
                cwd=self.build_dir,
                stdout=f_std,
                stderr=f_err,
            )

        logger.info("oxDNA binary rebuilt")

    def cleanup_build(self) -> None:
        """Clean up the build directory if it exists."""
        if self.build_dir.is_dir():
            shutil.rmtree(self.build_dir)
