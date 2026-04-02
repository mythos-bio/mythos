"""Common data structures for simulator I/O."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import chex
import jax.numpy as jnp
import jax_md
from jax.tree_util import tree_map

from mythos.energy.utils import q_to_back_base, q_to_base_normal
from mythos.input.trajectory import _write_state
from mythos.utils.helpers import tree_concatenate
from mythos.utils.types import ARR_OR_SCALAR, Arr_Box, Vector3D


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
        box_size: Optional box size associated with each state in the
            trajectory.
        metadata: Optional metadata associated with each state in the
          trajectory. This must be a dictionary where each value is a numerical
          array whose first axis has length corresponding to number of states.
        temperature: Optional per-state temperature in kT (thermal energy in
          simulation units). Shape ``(n_states,)``. When present,
          ``beta = 1 / temperature`` can be used for reweighting. ``None``
          indicates that the simulation temperature is unknown.
    """

    box_size: Arr_Box | None = None
    temperature: jnp.ndarray | None = None
    metadata: dict[str, jnp.ndarray] | None = None

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

    def slice(self, key: int | slice | jnp.ndarray | list) -> "SimulatorTrajectory":
        """Slice the trajectory."""
        if isinstance(key, int):
            key = slice(key, key + 1)
        if not isinstance(key, slice):
            key = jnp.asarray(key)

        metadata = None if self.metadata is None else tree_map(lambda x: x[key, ...], self.metadata)
        box_size = None if self.box_size is None else self.box_size[key, ...]
        temperature = None if self.temperature is None else self.temperature[key, ...]

        return self.replace(
            center=self.center[key, ...],
            orientation=jax_md.rigid_body.Quaternion(
                vec=self.orientation.vec[key, ...],
            ),
            box_size=box_size,
            temperature=temperature,
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

    @classmethod
    def concat(cls, trajectories: list["SimulatorTrajectory"]) -> "SimulatorTrajectory":
        """Concatenate a list of SimulatorTrajectory instances."""
        if not trajectories:
            raise ValueError("Cannot concatenate an empty list of trajectories.")
        if len(trajectories) == 1:
            return trajectories[0]

        box_size = _concat_optional_field([t.box_size for t in trajectories], "box sizes")
        temperature = _concat_optional_field([t.temperature for t in trajectories], "temperatures")

        merged_metadata = _merge_metadata(
            [t.metadata for t in trajectories],
            [t.length() for t in trajectories],
        )

        return trajectories[0].replace(
            center=jnp.concatenate([t.center for t in trajectories], axis=0),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.concatenate([t.orientation.vec for t in trajectories], axis=0)
            ),
            box_size=box_size,
            temperature=temperature,
            metadata=merged_metadata,
        )

    def __add__(self, other: "SimulatorTrajectory") -> "SimulatorTrajectory":
        """Concatenate two trajectories."""
        return self.__class__.concat([self, other])

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
            box_size: The default box size in 3 dimensions to write to the file,
                if trajectory box_size is not available. defaults to (0,0,0).
        """
        with Path(filepath).open("w") as f:
            for i in range(self.length()):
                coms = self.center[i]
                bb_vecs = q_to_back_base(self.orientation[i])
                base_norms = q_to_base_normal(self.orientation[i])
                dummy_vels_angmom = jnp.zeros((coms.shape[0], 6))  # vels and angular momenta are not available
                state = jnp.hstack([coms, bb_vecs, base_norms, dummy_vels_angmom])
                box = self.box_size[i] if self.box_size is not None else box_size
                _write_state(f, time=float(i), energies=jnp.zeros(3), state=state, box_size=box)


def _concat_optional_field(
    values: list[jnp.ndarray | None],
    label: str,
) -> jnp.ndarray | None:
    """Concatenate an optional per-trajectory field along axis 0.

    Returns ``None`` when all entries are ``None``, raises when the entries
    are a mix of ``None`` and non-``None``, and concatenates otherwise.
    """
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError(f"Cannot concatenate, trajectories have incompatible {label}.")
    return jnp.concatenate(values, axis=0)


def _merge_metadata(
    metadata_list: list[dict[str, jnp.ndarray] | None],
    lengths: list[int],
) -> dict[str, jnp.ndarray] | None:
    """Merge a list of metadata dictionaries for SimulatorTrajectory concatenation.

    If a key is missing in some dictionaries, it is filled with NaNs of
    the same shape (excluding leading axis which is num_states) as the
    corresponding array in the dictionaries that do have it. If a key is
    present in multiple dictionaries the shapes must be consistent beyond
    the leading axis.
    """
    if all(not m for m in metadata_list):
        return None
    dicts = [m or {} for m in metadata_list]
    for key in {k for d in dicts for k in d}:
        # Collect present entries to validate shape consistency and determine fill shape
        present = [d[key] for d in dicts if key in d]
        shape = present[0].shape[1:]
        if any(p.shape[1:] != shape for p in present[1:]):
            raise ValueError(f"Metadata key '{key}' has mismatched shapes when adding trajectories.")
        # Fill missing entries with NaNs of the appropriate shape
        for d, length in zip(dicts, lengths, strict=True):
            d.setdefault(key, jnp.full((length, *shape), jnp.nan))
    return tree_concatenate(dicts)
