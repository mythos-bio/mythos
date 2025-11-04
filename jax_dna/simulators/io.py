"""Common data structures for simulator I/O."""

from pathlib import Path

import chex
import jax.numpy as jnp
import jax_md

from jax_dna.energy.utils import q_to_back_base, q_to_base_normal
from jax_dna.input.trajectory import _write_state


@chex.dataclass()
class SimulatorTrajectory:
    """A trajectory of a simulation run."""

    rigid_body: jax_md.rigid_body.RigidBody

    def slice(self, key: int | slice) -> "SimulatorTrajectory":
        """Slice the trajectory."""
        if isinstance(key, int):
            key = slice(key, key + 1)

        return self.replace(
            rigid_body=jax_md.rigid_body.RigidBody(
                center=self.rigid_body.center[key, ...],
                orientation=jax_md.rigid_body.Quaternion(
                    vec=self.rigid_body.orientation.vec[key, ...],
                ),
            )
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
        return self.replace(
            rigid_body=jax_md.rigid_body.RigidBody(
                center=jnp.concat(
                    [self.rigid_body.center, other.rigid_body.center],
                    axis=0,
                ),
                orientation=jax_md.rigid_body.Quaternion(
                    vec=jnp.concatenate([self.rigid_body.orientation.vec, other.rigid_body.orientation.vec], axis=0)
                ),
            )
        )

    def to_file(self, filepath: Path) -> None:
        """Write the trajectory to an oxDNA file.

        Note that the SimulatorTrajectory does not store several of the fields
        necessary to fully reconstruct an oxDNA trajectory file (e.g. times, box
        size, velocities, angular momenta, and energies). Thus, times are filled
        with a monotonic sequence, while the rest of these fields are filled
        with 0's. The resultant file can be used for inspection and
        visualization of non-time-dependent state-by-state spatial information
        only.
        """
        with Path(filepath).open("w") as f:
            for i in range(self.length()):
                coms = self.rigid_body.center[i]
                bb_vecs = q_to_back_base(self.rigid_body.orientation[i])
                base_norms = q_to_base_normal(self.rigid_body.orientation[i])
                dummy_vels_angmom = jnp.zeros((coms.shape[0], 6))  # vels and angular momenta are not available
                state = jnp.hstack([coms, bb_vecs, base_norms, dummy_vels_angmom])
                _write_state(f, time=float(i), energies=jnp.zeros(3), state=state)
