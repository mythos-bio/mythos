"""Utilities for the GROMACS simulator."""

import logging
import shutil
from pathlib import Path

import jax_md
import MDAnalysis
import numpy as np

import mythos.simulators.io as jd_sio
from mythos.input.gromacs_input import update_mdp_params
from mythos.utils.helpers import run_command

# MDAnalysis reads into Angstroms, but Gromacs parameters in nm
ANGSTROMS_TO_NM = 0.1
logger = logging.getLogger(__name__)


def read_trajectory_mdanalysis(topology_file: Path, trajectory_file: Path) -> jd_sio.SimulatorTrajectory:
    """Read a GROMACS trajectory using MDAnalysis.

    Args:
        topology_file: Path to the topology file (e.g., output.tpr).
        trajectory_file: Path to the trajectory file (e.g., output.trr).

    Returns:
        SimulatorTrajectory containing the rigid body trajectory data.
    """
    logger.debug("Loading trajectory from %s with topology %s", trajectory_file, topology_file)

    u = MDAnalysis.Universe(str(topology_file), str(trajectory_file))

    # Extract center of mass positions for each frame
    n_frames = len(u.trajectory)
    n_atoms = len(u.atoms)

    logger.debug("Trajectory contains %d frames with %d atoms", n_frames, n_atoms)

    # Skip the first frame (initial state) by starting from frame 1
    positions = np.stack([ts.positions.copy() for ts in u.trajectory[1:]]).astype(np.float64)
    box_sizes = np.stack([ts.dimensions[:3].copy() for ts in u.trajectory[1:]]).astype(np.float64)
    n_frames = n_frames - 1

    # Create quaternions (identity for now - GROMACS doesn't typically store orientations)
    quaternions = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        (n_frames, n_atoms, 1),
    )

    return jd_sio.SimulatorTrajectory(
        center=positions * ANGSTROMS_TO_NM,
        orientation=jax_md.rigid_body.Quaternion(vec=quaternions),
        box_size=box_sizes * ANGSTROMS_TO_NM,
    )


def preprocess_topology(
    input_dir: str | Path,
    params: dict | None = None,
    copy_to: Path | None = None,
    output_prefix: str = "preprocessed",
    output_mdp_name: str = "preprocessed.mdp",
    gromacs_binary: str | None = None,
    mdp_name: str = "md.mdp",
    topology_name: str = "topol.top",
    structure_name: str = "membrane.gro",
    index_name: str = "index.ndx",
    log_prefix: str = "topology_preprocess",
) -> None:
    """Preprocess a GROMACS topology.

    This function runs `gmx grompp` to preprocess the GROMACS topology, applying
    any parameter updates to the .mdp file as needed. The preprocessed topology
    will be saved with the specified output prefix. Optionally copies input
    files to a new location to avoid modifying originals.

    Args:
        input_dir: Directory containing the GROMACS input files.
        params: Optional dictionary of parameters to update in the .mdp file.
        copy_to: Optional directory to copy input files to before preprocessing.
        output_prefix: Prefix for the preprocessed topology (.top extension)
            and tpr files (.tpr extension).
        output_mdp_name: Name of the output .mdp file after replacements.
        gromacs_binary: Optional path to the GROMACS binary (defaults to 'gmx' in PATH).
        mdp_name: Name of the .mdp file in the input directory.
        topology_name: Name of the topology file in the input directory.
        structure_name: Name of the structure file in the input directory.
        index_name: Name of the index file in the input directory.
        log_prefix: Prefix for log messages.
    """
    input_dir = Path(input_dir)

    # pre-emptively check for the GROMACS binary before copying files
    gromacs_binary = gromacs_binary or shutil.which("gmx")
    if gromacs_binary is None or not Path(gromacs_binary).exists():
        raise FileNotFoundError(f"GROMACS binary not found or does not exist at: {gromacs_binary}")

    if copy_to is not None:
        # Copy the input directory to the specified location to avoid modifying original files
        copy_dir = Path(copy_to)
        shutil.copytree(input_dir, copy_dir)
        input_dir = copy_dir

    update_mdp_params(input_dir / mdp_name, params or {}, out_file=input_dir / output_mdp_name)
    cmd = [
        gromacs_binary,
        "grompp",
        "-p",
        topology_name,
        "-f",
        output_mdp_name,
        "-c",
        structure_name,
        "-n",
        index_name,
        "-pp",
        f"{output_prefix}.top",
        "-o",
        f"{output_prefix}.tpr",
    ]
    run_command(cmd, cwd=input_dir, log_prefix=log_prefix)
