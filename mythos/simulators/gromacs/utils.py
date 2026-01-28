"""Utilities for the GROMACS simulator."""

import logging
from pathlib import Path

import jax_md
import MDAnalysis
import numpy as np

import mythos.simulators.io as jd_sio

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
    positions = np.stack([ts.positions for ts in u.trajectory[1:]]).astype(np.float64)
    n_frames = n_frames - 1

    # Create quaternions (identity for now - GROMACS doesn't typically store orientations)
    quaternions = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        (n_frames, n_atoms, 1),
    )

    return jd_sio.SimulatorTrajectory(
        center=positions,
        orientation=jax_md.rigid_body.Quaternion(vec=quaternions),
    )
