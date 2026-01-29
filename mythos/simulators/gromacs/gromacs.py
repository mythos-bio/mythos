"""GROMACS simulator module."""

import logging
import shutil
import typing
from dataclasses import field
from pathlib import Path

import chex
import numpy as np

from mythos.energy.base import EnergyFunction
from mythos.input.gromacs_input import update_mdp_params
from mythos.simulators import io as jd_sio
from mythos.simulators.base import InputDirSimulator, SimulatorOutput
from mythos.simulators.gromacs import utils as gromacs_utils
from mythos.utils.helpers import run_command

logger = logging.getLogger(__name__)


@chex.dataclass(frozen=True, kw_only=True)
class GromacsSimulator(InputDirSimulator):
    """A simulator based on running a GROMACS simulation.

    This simulator runs a GROMACS simulation from an input directory containing
    the necessary configuration files and outputs a trajectory in
    `SimulatorTrajectory` format. All _file parameters refer to filenames within
    the input directory.

    Arguments:
        input_dir: Path to the directory containing the GROMACS input files.
        energy_fn: The energy function to use for parameter updates. Parameters
            from this energy function will be used to update the topology file.
        mdp_file: Name of the .mdp (molecular dynamics parameter) file.
        topology_file: Name of the topology file (e.g., .top). Parameters from
            the energy function will be written to this file.
        trajectory_file: Name of the output trajectory file (e.g., .xtc, .trr).
        structure_file: Name of the structure/coordinate file (e.g., .gro, .pdb).
        binary_path: Path to the GROMACS binary. If not provided, will search
            for 'gmx' in PATH.
        input_overrides: Key-value pairs to override in the .mdp input file.
        overwrite_input: Whether to overwrite the input directory or copy it.
    """

    energy_fn: EnergyFunction
    mdp_file: str = "md.mdp"
    topology_file: str = "topol.top"
    structure_file: str = "membrane.gro"
    index_file: str = "index.ndx"
    binary_path: Path | None = None
    input_overrides: dict[str, typing.Any] = field(default_factory=dict)

    def __post_init__(self, *args, **kwds) -> None:
        """Check the validity of the configuration."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        for file in [self.mdp_file, self.topology_file, self.structure_file, self.index_file]:
            if not (self.input_dir / file).exists():
                raise FileNotFoundError(f"Required input file '{file}' not found in {self.input_dir}")

    def run_simulation(
        self,
        input_dir: Path,
        opt_params: dict[str, typing.Any] | None = None,
        seed: int | None = None,
        **_,
    ) -> SimulatorOutput:
        """Run a GROMACS simulation.

        Args:
            input_dir: Path to the working directory for the simulation.
            opt_params: Optional parameters to update. These will be used to
                update the energy function and subsequently the topology file.
            seed: Optional random seed for the simulation. By default, a random
                seed is generated.

        Returns:
            SimulatorOutput containing the trajectory.
        """
        mdp_path = input_dir / self.mdp_file
        topology_path = input_dir / self.topology_file
        gmx_binary = self.binary_path or shutil.which("gmx")
        if gmx_binary is None:
            raise FileNotFoundError(
                "GROMACS binary not found. Please install GROMACS into PATH or provide the path "
                "to the binary via the 'binary_path' argument."
            )

        seed = seed or np.random.default_rng().integers(0, 2**31)
        update_mdp_params(mdp_path, {**self.input_overrides, "gen-seed": seed})

        # Update topology file with energy function parameters and overrides
        self._update_topology_params(opt_params or {})

        logger.info("Starting GROMACS simulation")
        # prepare the run
        cmd = [
            gmx_binary,
            "grompp",
            "-f",
            f"{mdp_path}",
            "-c",
            f"{input_dir / self.structure_file}",
            "-p",
            f"{topology_path}",
            "-n",
            f"{input_dir / self.index_file}",
            "-o",
            "output.tpr",
        ]
        run_command(cmd, cwd=input_dir, log_prefix="grompp")

        # run the simulation
        cmd = [
            gmx_binary,
            "mdrun",
            "-deffnm",
            "output",
            "-ntmpi",
            "1",
            "-rdd",
            "1.5",
        ]
        run_command(cmd, cwd=input_dir, log_prefix="mdrun")
        logger.info("GROMACS simulation complete")

        return SimulatorOutput(observables=[self._read_trajectory(input_dir)])

    def _read_trajectory(self, input_dir: Path) -> jd_sio.SimulatorTrajectory:
        trajectory = gromacs_utils.read_trajectory_mdanalysis(
            topology_file=input_dir / "output.tpr",
            trajectory_file=input_dir / "output.trr",
        )

        logger.debug("GROMACS trajectory size: %s", trajectory.length())

        return trajectory

    def _update_topology_params(self, opt_params: dict[str, typing.Any]) -> None:
        # Parameter mapping from energy function to topology will be implemented
        # once the GROMACS energy function is defined. This will involve getting
        # updated params from self.energy_fn.with_params() and mapping them to
        # the GROMACS topology format.
        logger.debug("Topology parameter update not yet implemented for params: %s", opt_params)
