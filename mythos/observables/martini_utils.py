"""General utility functions for working with Martini observables."""
import MDAnalysis
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

from mythos.simulators.io import SimulatorTrajectory

NM_TO_ANGSTROMS = 10.0


def universe_from_trajectory(topology: MDAnalysis.Universe, trajectory: SimulatorTrajectory) -> MDAnalysis.Universe:
    """Reconstruct an MDAnalysis Universe from a topology and SimulatorTrajectory.

    This is useful in cases where we want to work against a Universe object
    within an objective function. The topology of the system (e.g. as read by
    the tpr file) will typically be static and can be passed as a property of
    the observable, then have the trajectory embedded to produce a new Universe
    object. The SimulatorTrajectory stores positions in nm, MDAnalysis expects
    Angstroms.  Box dimensions are extended with 90 degree angles for MDAnalysis
    compatibility.
    """
    positions = np.asarray(trajectory.center * NM_TO_ANGSTROMS)
    # box_size is (n_frames, 3) need (n_frames, 6) with 90 degree angles which
    # have been stripped from the trajectory when reading and converting to a
    # SimulatorTrajectory object. See mythos.simulators.io._read_gromacs_trajectory
    # for details.
    box3 = np.asarray(trajectory.box_size * NM_TO_ANGSTROMS)
    angles = np.broadcast_to(np.array([90.0, 90.0, 90.0], dtype=box3.dtype), box3.shape)
    dimensions = np.concatenate([box3, angles], axis=-1)  # (n_frames, 6)

    new_universe = topology.copy()
    new_universe.load_new(positions, format=MemoryReader, dimensions=dimensions, order="fac")
    return new_universe
