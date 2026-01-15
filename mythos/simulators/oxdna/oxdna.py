"""OXDNA sampler module.

Run an mythos simulation using an oxDNA sampler.
"""

import logging
import os
import shutil
import typing
from dataclasses import field
from pathlib import Path

import chex
import numpy as np

import mythos.input.oxdna_input as jd_oxdna
import mythos.simulators.io as jd_sio
import mythos.simulators.oxdna.utils as oxdna_utils
from mythos.energy.base import EnergyFunction
from mythos.simulators.base import InputDirSimulator, SimulatorOutput
from mythos.utils.helpers import run_command
from mythos.utils.types import Params, oxDNASimulatorType

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


@chex.dataclass(frozen=True, kw_only=True)
class oxDNASimulator(InputDirSimulator):  # noqa: N801 oxDNA is a special word
    """A sampler base on running an oxDNA simulation.

    This simulator runs an oxDNA simulation in a subprocess, first compiling
    oxDNA from source with the provided parameters, or by using a precompiled
    binary (in the case parameter updates are not desired).

    Arguments:
        input_dir: Path to the directory containing the oxDNA input file.
        sim_type: The type of oxDNA simulation to run.
        energy_fn: The energy function to use for default parameter updates.
        n_build_threads: Number of threads to use when building oxDNA from
            source.
        logger_config: Configuration for the logger.
        binary_path: Path to a precompiled oxDNA binary to use. This is mutually
            exclusive with source_path. When provided, the binary will be called
            and no recompilation will be performed. In such a case, parameters
            cannot be updated, and if supplied to the run will result in an
            error unless ignore_params is set to True.
        source_path: Path to the oxDNA source code to compile. Updating
            parameters in this simulator requires compiling oxDNA from source
            with the parameters built into the object code.
        ignore_params: Whether to ignore provided parameters when running the
            simulation. This argument is required to be True if there is no
            source_path set and parameters are passed.
        overwrite_input: Whether to overwrite the input directory or copy it. If
            this is False (default), the contents of the input_dir will be
            copied to a temporary directory for running the simulation to avoid
            overwriting input.
        input_overrides: Key-value pairs to override in the input file. The
            values accept scalar values that can be converted to str. For
            example: {"T": "275K", "steps": 10000}. WARNING: no validation is
            performed on the provided key-value pairs.
    """

    sim_type: oxDNASimulatorType
    energy_fn: EnergyFunction
    n_build_threads: int = 4
    logger_config: dict[str, typing.Any] | None = None
    binary_path: Path | None = None
    source_path: Path | None = None
    ignore_params: bool = False
    input_overrides: dict[str, typing.Any] = field(default_factory=dict)

    def __post_init__(self, *args, **kwds) -> None:
        """Check the validity of the configuration."""
        if not (bool(self.binary_path) ^ bool(self.source_path)):
            raise ValueError("Must set one and only one of binary_path or source_path")
        if not (Path(self.input_dir) / "input").exists():
            raise FileNotFoundError(f"Input file not found at: {self.input_dir / 'input'}")

    def with_cached_build(self, binary_path: Path) -> None:
        """Switch to use a precompiled binary.

        This may be useful when running on a cluster with a shared file system,
        or running on a single machine, particularly in cases where:
            N_simulators * n_build_threads > N_cpu_cores.

        Caution: the user is responsible for ensuring that the binary at
        provided path is pre-built for the appropriate parameter set, there is
        no check performed at simulation run-time to verify this.
        """
        return self.replace(binary_path=binary_path, source_path=None, ignore_params=True)

    def run_simulation(
        self, input_dir: Path, opt_params: Params|None = None, seed: float|None = None, **_
    ) -> SimulatorOutput:
        """Run an oxDNA simulation."""
        input_config = jd_oxdna.read(input_dir / "input")
        input_config.update(self.input_overrides)
        input_config["seed"] = seed or np.random.default_rng().integers(0, 2**32)
        jd_oxdna.write(input_config, input_dir / "input")

        if opt_params is not None:
            if self.source_path:
                self.build(input_dir = input_dir, new_params=opt_params, input_config=input_config)
            elif not self.ignore_params:
                raise ValueError("params provided without source_path. Set ignore_params to override")
        elif self.source_path:
            self.build(input_dir = input_dir, new_params={}, input_config=input_config)
        binary_path = self.binary_path or input_dir / "oxdna-build" / "bin" / "oxDNA"

        # remove existing trajectory and energy files (others?), otherwise they
        # will be appended to
        for output in ["trajectory_file", "energy_file"]:
            if file := input_config.get(output, None):
                input_dir.joinpath(file).unlink(missing_ok=True)

        logger.info("Starting oxDNA simulation")
        cmd = [binary_path, "input"]
        logger.debug("running command: %s", cmd)
        run_command(cmd, cwd=input_dir, log_prefix="oxdna")
        logger.info("oxDNA simulation complete")

        return SimulatorOutput(observables=[self._read_trajectory(input_dir)])


    def _read_trajectory(self, input_dir: Path) -> jd_sio.SimulatorTrajectory:
        trajectory = oxdna_utils.read_output_trajectory(input_file=input_dir / "input")

        logger.debug("oxDNA trajectory com size: %s", trajectory.state_rigid_body.center.shape)

        return jd_sio.SimulatorTrajectory(
            rigid_body=trajectory.state_rigid_body,
        )


    def build(self, *, input_dir: Path, new_params: Params, input_config: dict|None = None) -> None:
        """Update the simulation.

        This function will recompile the oxDNA binary with the new parameters.
        """
        cmake_bin = _guess_binary_location("cmake", CMAKE_BIN_ENV_VAR)
        make_bin = _guess_binary_location("make", MAKE_BIN_ENV_VAR)

        build_dir = input_dir / "oxdna-build"
        logger.info("Updating oxDNA parameters (build path: %s)", str(build_dir))

        build_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("build_dir: %s", build_dir)

        model_h = build_dir / "model.h"
        if not model_h.exists():
            model_h.write_text(Path(self.source_path).joinpath("src/model.h").read_text())

        updated_params = self.energy_fn.with_params(new_params).params_dict(exclude_non_optimizable=True)
        oxdna_utils.update_params(model_h, updated_params)

        input_config = input_config or jd_oxdna.read(input_dir / "input")

        if not (build_dir / "CMakeLists.txt").exists():
            cmd = [cmake_bin, self.source_path, f"-DCMAKE_CXX_FLAGS=--include {model_h}"]
            if input_config["backend"] == "CUDA":
                cmd = [*cmd, "-DCUDA=ON", "-DCUDA_COMMON_ARCH=OFF"]
            logger.debug("Attempting cmake using: %s", cmd)
            run_command(cmd, cwd=build_dir, log_prefix="oxdna.cmake")
            logger.debug("cmake completed")

        # rebuild the binary
        logger.debug("running make with %d processes", self.n_build_threads)
        run_command(
            [make_bin, f"-j{self.n_build_threads}", "clean", "oxDNA"],  # clean since model.h is not tracked
            cwd=build_dir,
            log_prefix="oxdna.make",
        )
        logger.info("oxDNA binary rebuilt")
