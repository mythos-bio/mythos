import jax.numpy as jnp
import jax_md
import pytest

import mythos.simulators.io as jd_sio
from mythos.input import trajectory


@pytest.mark.parametrize(
    ("n", "key", "expected_n"),
    [
        (10, 5, 1),
        (10, slice(5), 5),
    ],
)
def test_simulatortrajectory_slice(
    n: int,
    key: int | slice,
    expected_n: int,
) -> None:
    """Test the slice method of the SimulatorTrajectory class."""
    traj = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.zeros((n, 3)),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.zeros((n, 4)),
            ),
        )
    )

    sliced_traj = traj.slice(key)

    assert sliced_traj.rigid_body.center.shape == (expected_n, 3)
    assert sliced_traj.rigid_body.orientation.vec.shape == (expected_n, 4)


@pytest.mark.parametrize(
    ("n"),
    [(10), (1)],
)
def test_simulatortrajectory_length(
    n: int,
) -> None:
    """Test the length method of the SimulatorTrajectory class."""

    traj = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.zeros((n, 3)),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.zeros((n, 4)),
            ),
        )
    )

    assert traj.length() == n


@pytest.mark.parametrize(
    ("n_states", "n_nucleotides"),
    [
        (4, 10),
        (2, 5),
    ],
)
def test_simulator_trajectory_to_file(tmp_path, n_states, n_nucleotides):
    traj = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.ones((n_states, n_nucleotides, 3)),
            orientation=jax_md.rigid_body.Quaternion(
                # quarternions of (1,0,0,0) translate to back base and base
                # norms of (0, 0, 0), and vice-a-versa
                vec=jnp.dstack([jnp.ones((n_states, n_nucleotides, 1)), jnp.zeros((n_states, n_nucleotides, 3))]),
            ),
        )
    )
    output = tmp_path / "test.traj"
    traj.to_file(output)

    read_back = trajectory.from_file(output, strand_lengths=[n_nucleotides], is_5p_3p=False)
    assert jnp.allclose(read_back.state_rigid_body.center, traj.rigid_body.center)
    assert jnp.allclose(read_back.state_rigid_body.orientation.vec, traj.rigid_body.orientation.vec)
