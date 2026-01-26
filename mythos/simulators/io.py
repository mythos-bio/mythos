"""Common data structures for simulator I/O."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import jax_md
from jax import tree_map

from mythos.energy.utils import q_to_back_base, q_to_base_normal
from mythos.input.trajectory import _write_state
from mythos.utils.types import ARR_OR_SCALAR, Vector3D


@chex.dataclass()
class SimulatorTrajectory:
    """A trajectory of a simulation run.

    Parameters:
        rigid_body: The jax_md RigidBody representation of the trajectory.
        metadata: Optional metadata associated with each state in the
          trajectory. This must be a dictionary where each value is a numerical
          array of length equal to the number of states in the trajectory.
    """
    rigid_body: jax_md.rigid_body.RigidBody
    metadata: dict[str, jnp.ndarray]|None = None

    def with_state_metadata(self, **metadata: dict[str, ARR_OR_SCALAR]) -> "SimulatorTrajectory":
        """Set the same metadata for all states in the trajectory."""
        new_metadata = self.metadata.copy() if self.metadata is not None else {}
        for key, value in metadata.items():
            new_metadata[key] = jnp.stack([jnp.asarray(value)] * self.length())
        return self.replace(metadata=new_metadata)

    def filter(self, filter_fn: Callable[[Any], bool]) -> "SimulatorTrajectory":
        """Filter the trajectory based on metadata.

        Args:
            filter_fn: A function that takes in metadata tree and returns a
                boolean array of length equal to the number of states,
                indicating which states to keep.

        Returns:
            A new SimulatorTrajectory with only the states that pass the filter.
        """
        indices = jnp.where(filter_fn(self.metadata))[0]
        return self.slice(indices)

    def slice(self, key: int | slice | jnp.ndarray | list) -> "SimulatorTrajectory":
        """Slice the trajectory."""
        if isinstance(key, int):
            key = slice(key, key + 1)
        if not isinstance(key, slice):
            key = jnp.asarray(key)

        metadata = None if self.metadata is None else tree_map(lambda x: x[key], self.metadata)

        return self.replace(
            rigid_body=jax_md.rigid_body.RigidBody(
                center=self.rigid_body.center[key, ...],
                orientation=jax_md.rigid_body.Quaternion(
                    vec=self.rigid_body.orientation.vec[key, ...],
                ),
            ),
            metadata=metadata,
        )

    def length(self) -> int:
        """Return the length of the trajectory.

        Note, that this may have been more natural to implement as the built-in
        __len__ method. However, the chex.dataclass decorator overrides that
        method to be compatabile with the abc.Mapping interface

        See here:
        https://github.com/google-deepmind/chex/blob/8af2c9e8a19f3a57d9bd283c2a34148aef952f60/chex/_src/dataclass.py#L50
        """
        return self.rigid_body.center.shape[0]

    def __add__(self, other: "SimulatorTrajectory") -> "SimulatorTrajectory":
        """Concatenate two trajectories."""
        left_metadata = self.metadata or {}
        right_metadata = other.metadata or {}
        keys = left_metadata.keys() | right_metadata.keys()
        if keys:
            for key in keys:
                left_metadata.setdefault(key, jnp.array([jnp.nan] * self.length()))
                right_metadata.setdefault(key, jnp.array([jnp.nan] * other.length()))
            metadata = tree_map(lambda ll, rl: jnp.concatenate([ll, rl], axis=0), left_metadata, right_metadata)
        else:
            metadata = None

        return self.replace(
            rigid_body=jax_md.rigid_body.RigidBody(
                center=jnp.concat(
                    [self.rigid_body.center, other.rigid_body.center],
                    axis=0,
                ),
                orientation=jax_md.rigid_body.Quaternion(
                    vec=jnp.concatenate([self.rigid_body.orientation.vec, other.rigid_body.orientation.vec], axis=0)
                ),
            ),
            metadata=metadata,
        )

    def to_file(self, filepath: Path, box_size: Vector3D = (0, 0, 0)) -> None:
        """Write the trajectory to an oxDNA file.

        Note that the SimulatorTrajectory does not store several of the fields
        necessary to fully reconstruct an oxDNA trajectory file (e.g. times, box
        size, velocities, angular momenta, and energies). Thus, times are filled
        with a monotonic sequence, while the rest of these fields are filled
        with 0's. The resultant file can be used for inspection and
        visualization of non-time-dependent state-by-state spatial information
        only.

        Args:
            filepath: The path to write the trajectory file to.
            box_size: The box size in 3 dimensions to write to the file. defaults to (0,0,0).
        """
        with Path(filepath).open("w") as f:
            for i in range(self.length()):
                coms = self.rigid_body.center[i]
                bb_vecs = q_to_back_base(self.rigid_body.orientation[i])
                base_norms = q_to_base_normal(self.rigid_body.orientation[i])
                dummy_vels_angmom = jnp.zeros((coms.shape[0], 6))  # vels and angular momenta are not available
                state = jnp.hstack([coms, bb_vecs, base_norms, dummy_vels_angmom])
                _write_state(f, time=float(i), energies=jnp.zeros(3), state=state, box_size=box_size)
