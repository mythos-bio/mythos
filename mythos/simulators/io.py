"""Common data structures for simulator I/O."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import jax_md
from jax import tree_map
from typing_extensions import override

from mythos.energy.utils import q_to_back_base, q_to_base_normal
from mythos.input.trajectory import _write_state
from mythos.utils.helpers import tree_concatenate
from mythos.utils.types import ARR_OR_SCALAR, Vector3D


@chex.dataclass(frozen=True)
class SimulatorTrajectory(jax_md.rigid_body.RigidBody):
    """A trajectory of a simulation run.

    This class extends jax_md.rigid_body.RigidBody to include optional
    metadata associated with each state in the trajectory. This object can also
    store data associated with a single state, but in such a case certain
    methods do not make sense (e.g. filtering or slicing). Such single-state
    usage is primarily intended for use within mapping functions.

    Parameters:
        center: The center of mass positions for each rigid body at each
            state in the trajectory.
        orientation: The orientations (as quaternions) for each rigid body at
            each state in the trajectory.
        metadata: Optional metadata associated with each state in the
          trajectory. This must be a dictionary where each value is a numerical
          array whose first axis has length corresponding to number of states.
    """
    metadata: dict[str, jnp.ndarray]|None = None

    @classmethod
    def from_rigid_body(cls, rigid_body: jax_md.rigid_body.RigidBody, **kwargs: Any) -> "SimulatorTrajectory":
        """Create a SimulatorTrajectory from a RigidBody instance.

        Args:
            rigid_body: The RigidBody instance to create the SimulatorTrajectory from.
            **kwargs: Additional keyword arguments to pass to the
            SimulatorTrajectory constructor.

        Returns:
            A SimulatorTrajectory instance.
        """
        return cls(center=rigid_body.center, orientation=rigid_body.orientation, **kwargs)

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

    @override
    def __getitem__(self, idx: int | slice | jnp.ndarray | list) -> "SimulatorTrajectory":
        return self.slice(idx)

    def slice(self, key: int | slice | jnp.ndarray | list) -> "SimulatorTrajectory":
        """Slice the trajectory."""
        if isinstance(key, int):
            key = slice(key, key + 1)
        if not isinstance(key, slice):
            key = jnp.asarray(key)

        metadata = None if self.metadata is None else tree_map(lambda x: x[key, ...], self.metadata)

        return self.replace(
            center=self.center[key, ...],
            orientation=jax_md.rigid_body.Quaternion(
                vec=self.orientation.vec[key, ...],
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
        return self.center.shape[0]

    def __add__(self, other: "SimulatorTrajectory") -> "SimulatorTrajectory":
        """Concatenate two trajectories."""
        return self.replace(
            center=jnp.concat(
                [self.center, other.center],
                axis=0,
            ),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.concatenate([self.orientation.vec, other.orientation.vec], axis=0)
            ),
            metadata=_merge_metadata(self.metadata, self.length(), other.metadata, other.length()),
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
                coms = self.center[i]
                bb_vecs = q_to_back_base(self.orientation[i])
                base_norms = q_to_base_normal(self.orientation[i])
                dummy_vels_angmom = jnp.zeros((coms.shape[0], 6))  # vels and angular momenta are not available
                state = jnp.hstack([coms, bb_vecs, base_norms, dummy_vels_angmom])
                _write_state(f, time=float(i), energies=jnp.zeros(3), state=state, box_size=box_size)


def _merge_metadata(
        left: dict[str, jnp.ndarray]|None,
        len_left: int,
        right: dict[str, jnp.ndarray]|None,
        len_right: int,
    ) -> dict[str, jnp.ndarray]|None:
    """Merge two metadata dictionaries for SimulatorTrajectory concatenation.

    If a key is missing in one of the dictionaries, it is filled with NaNs of
    the same shape (excluding leading axis which is num_states) as the
    corresponding array in the other dictionary. If a key is present in both
    dictionaries the shapes must be consistent beyond the leading axis.
    """
    if not left and not right:
        return None
    left, right = (left or {}, right or {})
    for key in left.keys() | right.keys():
        if key in left and key in right and left[key].shape[1:] != right[key].shape[1:]:
            raise ValueError(f"Metadata key '{key}' has mismatched shapes when adding trajectories.")
        shape = left.get(key, right.get(key)).shape[1:]
        # fill with NaNs of the appropriate shape where missing.
        left.setdefault(key, jnp.full((len_left, *shape), jnp.nan))
        right.setdefault(key, jnp.full((len_right, *shape), jnp.nan))
    return tree_concatenate([left, right])
