"""Tests for SimpleOptimizer."""

from collections.abc import Callable
from dataclasses import field

import chex
import jax.numpy as jnp
import optax
import pytest
from typing_extensions import override

import mythos.optimization.objective as jdna_objective
import mythos.optimization.optimization as opt
from mythos.simulators.base import Simulator, SimulatorOutput


@chex.dataclass(frozen=True, kw_only=True)
class MockSimulator(Simulator):
    """A mock simulator for testing."""

    name: str = "mock_simulator"
    return_observables: list[jnp.ndarray]

    def run(self, *_args, call_count=0, **_kwargs) -> SimulatorOutput:
        return SimulatorOutput(
            observables=self.return_observables,
            state={"call_count": call_count + 1},
        )


@chex.dataclass(frozen=True, kw_only=True)
class MockObjective(jdna_objective.Objective):
    """A mock objective for testing with configurable readiness."""

    readiness_function: Callable = field(default=lambda _: True)

    def calculate(self, observables, opt_params, call_count=0, **_kwargs):
        obs = next(iter(observables.values()))
        grads = {k: v*0.1 for k, v in opt_params.items()}
        return jdna_objective.ObjectiveOutput(
            is_ready=self.readiness_function(call_count),
            state={"call_count": call_count + 1},
            observables={"total": obs.sum()},
            grads=grads,
        )


class TestSimpleOptimizerStep:
    """Tests for SimpleOptimizer.step method."""

    def test_step_initializes_empty_state_when_none_provided(self):
        """Test that step initializes an empty state when none is provided."""
        objective = MockObjective(
            name="test",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
        )
        simulator = MockSimulator(return_observables=[jnp.array([1.0, 2.0, 3.0])])
        optimizer = optax.sgd(0.01)

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = simple_opt.step(params, state=None)

        assert output.opt_params is not None
        assert output.state is not None
        assert output.grads is not None

    def test_step_runs_simulation_when_objective_not_ready(self):
        """Test that step runs simulation when objective returns is_ready=False."""
        # First call (call_count=0) returns not ready, second call (call_count=1) returns ready
        objective = MockObjective(
            name="test",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
            readiness_function=lambda count: count != 1,
        )
        simulator = MockSimulator(return_observables=[jnp.array([1.0, 2.0, 3.0])])
        optimizer = optax.sgd(0.01)

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = simple_opt.step(params)
        assert output.state.component_state[objective.name]["call_count"] == 1

        # on the second step, we exercise the simulator assuming it is no longer
        # ready, which means we should make another call
        output = simple_opt.step(params, state=output.state)
        assert output.state.component_state[objective.name]["call_count"] == 3
        assert output.state.component_state[simulator.name]["call_count"] == 2

    def test_step_skips_simulation_when_objective_ready(self):
        """Test that step skips simulation when objective is already ready."""
        objective = MockObjective(
            name="test",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
            readiness_function=lambda _: True,  # Always ready
        )
        simulator = MockSimulator(return_observables=[jnp.array([0.0, 0.0])])
        optimizer = optax.sgd(0.01)

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        # Provide state with existing observables
        initial_state = opt.OptimizerState(
            observables={"traj": jnp.array([1.0, 2.0])},
        )
        params = {"param": jnp.array(1.0)}
        output = simple_opt.step(params, state=initial_state)

        # Simulator should not have been called - no simulator metadata
        assert output.state.component_state.get(simulator.name) == {}
        assert output.observables == {"test": {"total": jnp.array(3.0)}}
        assert output.grads == {"param": jnp.array(0.1)}

    def test_step_updates_params_with_gradients(self):
        """Test that step applies gradients to update params."""
        objective = MockObjective(
            name="test",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
        )
        simulator = MockSimulator(return_observables=[jnp.array([1.0])])
        optimizer = optax.sgd(1.0)  # learning rate 1.0 for easy verification

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = simple_opt.step(params)

        # With SGD lr=1.0, new_param = param - lr * grad = 1.0 - 1.0 * 0.1 = 0.9
        assert jnp.allclose(output.opt_params["param"], jnp.array(0.9))


class TestSimpleOptimizerStatePassing:
    """Tests for state passing in SimpleOptimizer."""

    def test_objective_state_stored_in_optimizer_state(self):
        """Test that objective output state is stored in optimizer state."""
        objective = MockObjective(
            name="test_objective",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
        )
        simulator = MockSimulator(
            name="test_simulator",
            return_observables=[jnp.array([1.0, 2.0, 3.0])],
        )
        optimizer = optax.sgd(0.01)

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = simple_opt.step(params)

        # Objective state should be stored under objective name with call_count
        assert "test_objective" in output.state.component_state
        assert "call_count" in output.state.component_state["test_objective"]

    def test_simulator_state_stored_in_optimizer_state(self):
        """Test that simulator output state is stored in optimizer state."""
        objective = MockObjective(
            name="test_obj",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
        )
        simulator = MockSimulator(
            name="test_sim",
            return_observables=[jnp.array([1.0, 2.0, 3.0])],
        )
        optimizer = optax.sgd(0.01)

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = simple_opt.step(params)

        # Simulator state should be stored under simulator name with call_count
        assert "test_sim" in output.state.component_state
        assert output.state.component_state["test_sim"]["call_count"] == 1

    def test_state_passed_to_objective_compute(self):
        """Test that state from optimizer state is passed to objective.compute."""
        # Use a custom objective that tracks call_count from state
        call_counts_received = []

        @chex.dataclass(frozen=True, kw_only=True)
        class TrackingObjective(jdna_objective.Objective):
            @override
            def calculate(self, observables, call_count=0, **_kwargs):
                call_counts_received.append(call_count)
                return jdna_objective.ObjectiveOutput(
                    is_ready=True,
                    grads={"param": jnp.array(0.1)},
                )

        objective = TrackingObjective(
            name="tracking_objective",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
        )
        simulator = MockSimulator(return_observables=[jnp.array([1.0])])
        optimizer = optax.sgd(0.01)

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        # Create state with existing state containing call_count
        initial_state = opt.OptimizerState(
            observables={"traj": jnp.array([1.0])},
            component_state={"tracking_objective": {"call_count": 5}},
        )
        params = {"param": jnp.array(1.0)}
        simple_opt.step(params, state=initial_state)

        # Check that call_count was passed from state
        assert len(call_counts_received) == 1
        assert call_counts_received[0] == 5

    def test_state_preserved_across_steps(self):
        """Test that state is preserved and updated across multiple steps."""
        objective = MockObjective(
            name="counter_objective",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
        )
        simulator = MockSimulator(return_observables=[jnp.array([1.0])])
        optimizer = optax.sgd(0.01)

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        # First step
        initial_state = opt.OptimizerState(observables={"traj": jnp.array([1.0])})
        params = {"param": jnp.array(1.0)}
        output1 = simple_opt.step(params, state=initial_state)

        # call_count should be 1 after first step
        assert output1.state.component_state["counter_objective"]["call_count"] == 1

        # Second step using output state
        output2 = simple_opt.step(output1.opt_params, state=output1.state)

        # call_count should be 2 after second step
        assert output2.state.component_state["counter_objective"]["call_count"] == 2

class TestSimpleOptimizerOptimizerState:
    """Tests for optimizer state management."""

    def test_optimizer_state_initialized_on_first_step(self):
        """Test that optimizer state is initialized on first step."""
        objective = MockObjective(
            name="test",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
        )
        simulator = MockSimulator(return_observables=[jnp.array([1.0])])
        optimizer = optax.adam(0.01)

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = simple_opt.step(params)

        # Optimizer state should be initialized
        assert output.state.optimizer_state is not None

    def test_optimizer_state_preserved_across_steps(self):
        """Test that optimizer state is preserved and updated across steps."""
        objective = MockObjective(
            name="test",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
        )
        simulator = MockSimulator(return_observables=[jnp.array([1.0])])
        optimizer = optax.adam(0.01)

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output1 = simple_opt.step(params)

        # First optimizer state should exist
        assert output1.state.optimizer_state is not None

        # Second step
        output2 = simple_opt.step(output1.opt_params, state=output1.state)

        # States should be different (updated by optimizer)
        # For Adam, the step count should have incremented
        assert output2.state.optimizer_state is not None


class TestSimpleOptimizerErrorHandling:
    """Tests for error handling in SimpleOptimizer."""

    def test_raises_when_objective_never_ready(self):
        """Test that an error is raised if objective is never ready after simulation."""
        # Objective always returns not ready
        objective = MockObjective(
            name="test",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
            readiness_function=lambda _: False,  # Never ready
        )
        simulator = MockSimulator(return_observables=[jnp.array([1.0])])
        optimizer = optax.sgd(0.01)

        simple_opt = opt.SimpleOptimizer(
            objective=objective,
            simulator=simulator,
            optimizer=optimizer,
        )

        params = {"param": jnp.array(1.0)}

        with pytest.raises(ValueError, match="Objective readiness check failed"):
            simple_opt.step(params)
