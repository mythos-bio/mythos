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


def test_simulatortrajectory_with_state_metadata() -> None:
    """Test the with_state_metadata method sets metadata for all states."""
    n = 5
    traj = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.zeros((n, 3)),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.zeros((n, 4)),
            ),
        )
    )

    metadata = {"force": 10.0, "torque": 5.0}
    traj_with_metadata = traj.with_state_metadata(metadata)

    assert len(traj_with_metadata.metadata) == n
    for md in traj_with_metadata.metadata:
        assert md == metadata
        assert md["force"] == 10.0
        assert md["torque"] == 5.0


def test_simulatortrajectory_filter_basic() -> None:
    """Test the filter method with a simple predicate."""
    n = 10
    traj = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.arange(n * 3).reshape((n, 3)),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.zeros((n, 4)),
            ),
        ),
        metadata=[{"value": i} for i in range(n)],
    )

    # Filter for even values only
    filtered_traj = traj.filter(lambda md: md["value"] % 2 == 0)

    assert filtered_traj.length() == 5
    # Check that metadata is correctly filtered
    for md in filtered_traj.metadata:
        assert md["value"] % 2 == 0


def test_simulatortrajectory_filter_by_force_torque() -> None:
    """Test filtering trajectory by force/torque conditions (like mechanical sim)."""
    # Create trajectory with different force/torque conditions
    n_per_condition = 3
    conditions = [
        {"force": 0, "torque": 0},
        {"force": 2, "torque": 0},
        {"force": 2, "torque": 10},
        {"force": 2, "torque": 20},
    ]

    metadata = []
    for cond in conditions:
        metadata.extend([cond] * n_per_condition)

    total_n = len(metadata)
    traj = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.arange(total_n * 3).reshape((total_n, 3)),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.zeros((total_n, 4)),
            ),
        ),
        metadata=metadata,
    )

    # Filter stretch experiments (torque == 0)
    stretch_traj = traj.filter(lambda md: md["torque"] == 0)
    assert stretch_traj.length() == 6  # 2 conditions * 3 states

    # Filter torsion experiments (force == 2 and torque > 0)
    torsion_traj = traj.filter(lambda md: md["force"] == 2 and md["torque"] > 0)
    assert torsion_traj.length() == 6  # 2 conditions * 3 states

    # Filter specific condition
    specific_traj = traj.filter(lambda md: md["force"] == 2 and md["torque"] == 10)
    assert specific_traj.length() == 3


def test_simulatortrajectory_filter_empty_result() -> None:
    """Test filter that matches no states returns empty trajectory."""
    n = 5
    traj = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.zeros((n, 3)),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.zeros((n, 4)),
            ),
        ),
        metadata=[{"value": i} for i in range(n)],
    )

    # Filter that matches nothing
    filtered_traj = traj.filter(lambda md: md["value"] > 100)

    assert filtered_traj.length() == 0


def test_simulatortrajectory_filter_preserves_data() -> None:
    """Test that filter preserves rigid body data correctly."""
    n = 4
    centers = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=jnp.float32)
    traj = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=centers,
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.zeros((n, 4)),
            ),
        ),
        metadata=[{"keep": True}, {"keep": False}, {"keep": True}, {"keep": False}],
    )

    filtered_traj = traj.filter(lambda md: md["keep"])

    assert filtered_traj.length() == 2
    # Should have states 0 and 2
    assert jnp.allclose(filtered_traj.rigid_body.center[0], jnp.array([1, 2, 3]))
    assert jnp.allclose(filtered_traj.rigid_body.center[1], jnp.array([7, 8, 9]))

