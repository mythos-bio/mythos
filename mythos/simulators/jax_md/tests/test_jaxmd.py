"""Tests for the JaxMDSimulator."""

import chex
import jax
import jax.numpy as jnp
import jax_md
import jax_md.rigid_body
import jax_md.space
import numpy as np
import pytest

import mythos.energy.base as jd_energy_fn
import mythos.simulators.base as jd_sim_base
import mythos.simulators.jax_md.jaxmd as jaxmd_mod
import mythos.simulators.jax_md.utils as jaxmd_utils

N_NUCLEOTIDES = 4
N_STEPS = 3


@chex.dataclass(frozen=True, kw_only=True)
class MockEnergyFunction(jd_energy_fn.EnergyFunction):
    """A minimal mock energy function for testing."""

    eps: float = 1.0

    def __call__(self, body: jax_md.rigid_body.RigidBody) -> float:
        return jnp.sum(body.center) * self.eps

    def with_params(self, params):
        return MockEnergyFunction(eps=params.get("eps", self.eps))

    def with_props(self, **_kwargs):
        return self

    def with_noopt(self, *_params):
        return self

    def params_dict(self, **_kwargs):
        return {"eps": self.eps}

    def opt_params(self):
        return {"eps": self.eps}


@pytest.fixture
def mock_energy_fn():
    """Create a mock energy function."""
    return MockEnergyFunction()


@pytest.fixture
def simulator_params():
    """Create static simulator params."""
    return jaxmd_utils.StaticSimulatorParams(
        seq=jnp.zeros(N_NUCLEOTIDES, dtype=jnp.int32),
        mass=jax_md.rigid_body.RigidBody(
            center=jnp.ones((N_NUCLEOTIDES, 3)),
            orientation=jax_md.rigid_body.RigidBody(
                center=jnp.ones((N_NUCLEOTIDES, 3)),
                orientation=jnp.ones((N_NUCLEOTIDES, 3)),
            ),
        ),
        gamma=jax_md.rigid_body.RigidBody(
            center=jnp.ones((N_NUCLEOTIDES, 3)) * 0.1,
            orientation=jax_md.rigid_body.RigidBody(
                center=jnp.ones((N_NUCLEOTIDES, 3)) * 0.1,
                orientation=jnp.ones((N_NUCLEOTIDES, 3)) * 0.1,
            ),
        ),
        bonded_neighbors=jnp.array([[0, 1], [1, 2], [2, 3]]),
        checkpoint_every=0,
        dt=0.001,
        kT=1.0,
    )


@pytest.fixture
def neighbors():
    """Create a no-op neighbor helper."""
    return jaxmd_utils.NoNeighborList(
        unbonded_nbrs=np.array([[1, 2], [0, 3], [0, 3], [1, 2]]),
    )


@pytest.fixture
def space():
    """Create a free space."""
    return jax_md.space.free()


@chex.dataclass
class FakeSimState:
    """A minimal JAX-compatible simulation state."""

    position: jax_md.rigid_body.RigidBody
    mass: jax_md.rigid_body.RigidBody


def _make_fake_simulator_init():
    """Create a fake simulator_init function that mimics jax_md simulator init/step."""

    def simulator_init(energy_fn, shift_fn, **kwargs):
        def init_fn(key, R, **init_kwargs):  # noqa: N803 it is called by named parameters in the run function
            mass = jax_md.rigid_body.RigidBody(
                center=jnp.ones_like(R.center),
                orientation=jax_md.rigid_body.RigidBody(
                    center=jnp.ones_like(R.center),
                    orientation=jnp.ones_like(R.center),
                ),
            )
            return FakeSimState(position=R, mass=mass)

        def step_fn(state, **step_kwargs):
            new_center = state.position.center + 0.01
            new_position = jax_md.rigid_body.RigidBody(
                center=new_center,
                orientation=state.position.orientation,
            )
            return FakeSimState(position=new_position, mass=state.mass)

        return init_fn, step_fn

    return simulator_init


class TestJaxMDSimulatorInstantiation:
    """Tests for JaxMDSimulator construction."""

    def test_frozen_dataclass_run_is_set(self, mock_energy_fn, simulator_params, space, neighbors):
        """Test that __post_init__ properly sets self.run on a frozen dataclass."""
        sim = jaxmd_mod.JaxMDSimulator(
            energy_fn=mock_energy_fn,
            simulator_params=simulator_params,
            space=space,
            simulator_init=_make_fake_simulator_init(),
            neighbors=neighbors,
        )

        # The run attribute must be a callable, not the base class stub
        assert callable(sim.run)
        # Verify it's the built function, not the base Simulator.run
        assert sim.run is not jd_sim_base.Simulator.run

    def test_is_simulator_subclass(self, mock_energy_fn, simulator_params, space, neighbors):
        """Test JaxMDSimulator is a Simulator."""
        sim = jaxmd_mod.JaxMDSimulator(
            energy_fn=mock_energy_fn,
            simulator_params=simulator_params,
            space=space,
            simulator_init=_make_fake_simulator_init(),
            neighbors=neighbors,
        )
        assert isinstance(sim, jd_sim_base.Simulator)

    def test_fields_are_accessible(self, mock_energy_fn, simulator_params, space, neighbors):
        """Test that fields set during construction are accessible."""
        sim = jaxmd_mod.JaxMDSimulator(
            energy_fn=mock_energy_fn,
            simulator_params=simulator_params,
            space=space,
            simulator_init=_make_fake_simulator_init(),
            neighbors=neighbors,
        )
        assert sim.energy_fn is mock_energy_fn
        assert sim.simulator_params is simulator_params
        assert sim.neighbors is neighbors

    def test_frozen_prevents_mutation(self, mock_energy_fn, simulator_params, space, neighbors):
        """Test that the frozen dataclass prevents attribute mutation."""
        sim = jaxmd_mod.JaxMDSimulator(
            energy_fn=mock_energy_fn,
            simulator_params=simulator_params,
            space=space,
            simulator_init=_make_fake_simulator_init(),
            neighbors=neighbors,
        )
        with pytest.raises(AttributeError):
            sim.energy_fn = MockEnergyFunction(eps=2.0)


class TestBuildRunFn:
    """Tests for the build_run_fn function."""

    def test_returns_callable(self, mock_energy_fn, simulator_params, space, neighbors):
        """Test that build_run_fn returns a callable."""
        run_fn = jaxmd_mod.build_run_fn(
            energy_fn=mock_energy_fn,
            simulator_params=simulator_params,
            space=space,
            simulator_init=_make_fake_simulator_init(),
            neighbors=neighbors,
        )
        assert callable(run_fn)

    def test_run_fn_returns_simulator_output(self, mock_energy_fn, simulator_params, space, neighbors):
        """Test that the run function returns a SimulatorOutput."""
        run_fn = jaxmd_mod.build_run_fn(
            energy_fn=mock_energy_fn,
            simulator_params=simulator_params,
            space=space,
            simulator_init=_make_fake_simulator_init(),
            neighbors=neighbors,
        )

        init_state = jax_md.rigid_body.RigidBody(
            center=jnp.zeros((N_NUCLEOTIDES, 3)),
            orientation=jax_md.rigid_body.Quaternion(jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (N_NUCLEOTIDES, 1))),
        )

        key = jax.random.PRNGKey(0)
        result = run_fn({"eps": 1.0}, init_state, N_STEPS, key)

        assert isinstance(result, jd_sim_base.SimulatorOutput)
        assert len(result.observables) == 1

    def test_trajectory_has_correct_length(self, mock_energy_fn, simulator_params, space, neighbors):
        """Test that the output trajectory has the correct number of frames."""
        run_fn = jaxmd_mod.build_run_fn(
            energy_fn=mock_energy_fn,
            simulator_params=simulator_params,
            space=space,
            simulator_init=_make_fake_simulator_init(),
            neighbors=neighbors,
        )

        init_state = jax_md.rigid_body.RigidBody(
            center=jnp.zeros((N_NUCLEOTIDES, 3)),
            orientation=jax_md.rigid_body.Quaternion(jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (N_NUCLEOTIDES, 1))),
        )

        key = jax.random.PRNGKey(42)
        result = run_fn({"eps": 1.0}, init_state, N_STEPS, key)

        trajectory = result.observables[0]
        assert trajectory.center.shape[0] == N_STEPS

    def test_temperature_is_set_from_kt(self, mock_energy_fn, simulator_params, space, neighbors):
        """Test that the trajectory temperature matches the simulator kT."""
        run_fn = jaxmd_mod.build_run_fn(
            energy_fn=mock_energy_fn,
            simulator_params=simulator_params,
            space=space,
            simulator_init=_make_fake_simulator_init(),
            neighbors=neighbors,
        )

        init_state = jax_md.rigid_body.RigidBody(
            center=jnp.zeros((N_NUCLEOTIDES, 3)),
            orientation=jax_md.rigid_body.Quaternion(jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (N_NUCLEOTIDES, 1))),
        )

        key = jax.random.PRNGKey(0)
        result = run_fn({"eps": 1.0}, init_state, N_STEPS, key)

        trajectory = result.observables[0]
        np.testing.assert_allclose(trajectory.temperature, jnp.full(N_STEPS, simulator_params.kT))
