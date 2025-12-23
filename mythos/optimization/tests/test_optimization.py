"""Tests for optimization module."""

from collections.abc import Callable
from dataclasses import field

import chex
import jax.numpy as jnp
import optax
import pytest
from typing_extensions import override

import mythos.optimization.objective as jdna_objective
import mythos.optimization.optimization as jdna_optimization
from mythos.simulators.base import Simulator, SimulatorOutput


@chex.dataclass(frozen=True, kw_only=True)
class MockSimulator(Simulator):
    name: str = "mock_simulator"
    return_observables: list[jnp.ndarray] = field(default_factory=lambda: [jnp.array([1.0])])
    exposed_observables: list[str] = field(default_factory=lambda: ["trajectory"])
    output_state: dict = field(default_factory=dict)
    state_tracker: dict = field(default_factory=dict)  # external object to track state passed in

    @override
    def run(self, *_args, opt_params, **state) -> SimulatorOutput:
        self.state_tracker.update(state)
        return SimulatorOutput(
            observables=self.return_observables,
            state={"call_count": state.get("call_count", 0) + 1, **self.output_state, **state},
        )

    def exposes(self) -> list[str]:
        return self.exposed_observables


@chex.dataclass(frozen=True, kw_only=True)
class MockObjective(jdna_objective.Objective):
    readiness_function: Callable = field(default=lambda _: True)
    output_state: dict = field(default_factory=dict)
    output_observables: dict = field(default_factory=dict)
    state_tracker: dict = field(default_factory=dict)  # external object to track state passed in

    def compute(self, observables, opt_params, **state):
        self.state_tracker.update(state)
        obs = next(iter(observables.values()))
        grads = {k: v * 0.1 for k, v in opt_params.items()}
        return jdna_objective.ObjectiveOutput(
            is_ready=self.readiness_function(state.get("call_count", 0)),
            state={"call_count": state.get("call_count", 0) + 1, **self.output_state, **state},
            observables={"total": obs.sum(), **self.output_observables},
            grads=grads,
        )


@pytest.fixture
def basic_objective():
    return MockObjective(
        name="test_obj",
        required_observables=("trajectory",),
        grad_or_loss_fn=lambda x: x,
    )


@pytest.fixture
def basic_simulator():
    return MockSimulator(
        name="test_sim",
        return_observables=[jnp.array([1.0, 2.0, 3.0])],
    )


@pytest.fixture
def basic_optimizer():
    return optax.sgd(0.01)


@pytest.fixture
def basic_optimization(basic_objective, basic_simulator, basic_optimizer):
    return jdna_optimization.RayOptimizer(
        objectives=[basic_objective],
        simulators=[basic_simulator],
        aggregate_grad_fn=lambda grads: grads[0] if grads else {},
        optimizer=basic_optimizer,
    )


class MockRef:  # mock ray object ref, which wraps a value in hashable container
    def __init__(self, value):
        self.value = value


@pytest.fixture(autouse=True, params=[0, 3])
def _mock_ray(monkeypatch, request):  # mocks for ray infrastructure
    class StatefulWaiter:
        def __init__(self):
            self.call_count = -1
            self.wait_delay_count = request.param
        def __call__(self, refs, **_kwargs):
            self.call_count += 1
            if self.call_count <= self.wait_delay_count:  # used the ensure we visit some of ref filter dedup logic
                return [], list(refs)
            return list(refs), []

    def mock_ray_get(ref):
        return ref.value

    def mock_create_and_run_remote(_self, fun, ray_options, *args):
        result = fun(*args)
        # Handle multi-return case (num_returns > 1)
        num_returns = ray_options.get("num_returns", 1)
        if num_returns > 1 and isinstance(result, tuple):
            return [MockRef(r) for r in result]
        return MockRef(result)

    monkeypatch.setattr("mythos.optimization.optimization.ray.wait", StatefulWaiter())
    monkeypatch.setattr("mythos.optimization.optimization.ray.get", mock_ray_get)
    monkeypatch.setattr(
        jdna_optimization.RayOptimizer,
        "_create_and_run_remote",
        mock_create_and_run_remote,
    )


# =============================================================================
# Tests for __post_init__ validation
# =============================================================================


class TestOptimizationPostInit:

    def test_raises_when_no_objectives(self, basic_simulator, basic_optimizer):
        with pytest.raises(ValueError, match=jdna_optimization.ERR_MISSING_OBJECTIVES):
            jdna_optimization.RayOptimizer(
                objectives=[],
                simulators=[basic_simulator],
                aggregate_grad_fn=lambda x: x,
                optimizer=basic_optimizer,
            )

    def test_raises_when_no_simulators(self, basic_objective, basic_optimizer):
        with pytest.raises(ValueError, match=jdna_optimization.ERR_MISSING_SIMULATORS):
            jdna_optimization.RayOptimizer(
                objectives=[basic_objective],
                simulators=[],
                aggregate_grad_fn=lambda x: x,
                optimizer=basic_optimizer,
            )

    def test_raises_when_no_aggregate_grad_fn(self, basic_objective, basic_simulator, basic_optimizer):
        with pytest.raises(ValueError, match=jdna_optimization.ERR_MISSING_AGG_GRAD_FN):
            jdna_optimization.RayOptimizer(
                objectives=[basic_objective],
                simulators=[basic_simulator],
                aggregate_grad_fn=None,
                optimizer=basic_optimizer,
            )

    def test_raises_when_no_optimizer(self, basic_objective, basic_simulator):
        with pytest.raises(ValueError, match=jdna_optimization.ERR_MISSING_OPTIMIZER):
            jdna_optimization.RayOptimizer(
                objectives=[basic_objective],
                simulators=[basic_simulator],
                aggregate_grad_fn=lambda x: x,
                optimizer=None,
            )

    def test_raises_when_duplicate_names(self, basic_optimizer):
        objective = MockObjective(
            name="duplicate_name",
            required_observables=("traj",),
            grad_or_loss_fn=lambda x: x,
        )
        simulator = MockSimulator(name="duplicate_name")
        with pytest.raises(ValueError, match="unique"):
            jdna_optimization.RayOptimizer(
                objectives=[objective],
                simulators=[simulator],
                aggregate_grad_fn=lambda x: x,
                optimizer=basic_optimizer,
            )

    def test_raises_when_duplicate_exposes(self, basic_objective, basic_optimizer):
        simulator1 = MockSimulator(name="sim_1", exposed_observables=["shared_obs"])
        simulator2 = MockSimulator(name="sim_1", exposed_observables=["shared_obs"])  # Same name = same exposes
        with pytest.raises(ValueError, match="unique"):
            jdna_optimization.RayOptimizer(
                objectives=[basic_objective],
                simulators=[simulator1, simulator2],
                aggregate_grad_fn=lambda x: x,
                optimizer=basic_optimizer,
            )

    def test_raises_when_exposes_and_name_clash(self, basic_objective, basic_optimizer):
        simulator1 = MockSimulator(name="sim_1", exposed_observables=["sim_1"])
        with pytest.raises(ValueError, match="unique"):
            jdna_optimization.RayOptimizer(
                objectives=[basic_objective],
                simulators=[simulator1],
                aggregate_grad_fn=lambda x: x,
                optimizer=basic_optimizer,
            )


class TestOptimizationStep:

    def test_step_returns_optimizer_output(
        self, basic_objective, basic_simulator, basic_optimizer
    ):
        opt = jdna_optimization.RayOptimizer(
            objectives=[basic_objective],
            simulators=[basic_simulator],
            aggregate_grad_fn=lambda grads: grads[0] if grads else {},
            optimizer=basic_optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = opt.step(params=params)

        assert output is not None
        assert isinstance(output, jdna_optimization.OptimizerOutput)
        assert output.opt_params is not None
        assert output.state is not None
        assert output.grads is not None
        assert output.observables is not None

    def test_step_calls_simulator_with_state(
        self, basic_objective, basic_optimizer
    ):
        received_state = {}

        simulator = MockSimulator(name="test_sim", state_tracker=received_state)
        opt = jdna_optimization.RayOptimizer(
            objectives=[basic_objective],
            simulators=[simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=basic_optimizer,
        )

        initial_state = jdna_optimization.OptimizerState(
            component_state={"test_sim": {"custom_key": "custom_value", "call_count": 5}},
        )
        params = {"param": jnp.array(1.0)}
        opt.step(params=params, state=initial_state)

        assert received_state["custom_key"] == "custom_value"
        assert received_state["call_count"] == 5

    def test_step_calls_objective_with_state(
        self, basic_simulator, basic_optimizer
    ):
        received_state = {}

        objective = MockObjective(
            name="test_obj",
            required_observables=("trajectory",),
            grad_or_loss_fn=lambda x: x,
            state_tracker=received_state,
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[objective],
            simulators=[basic_simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=basic_optimizer,
        )

        initial_state = jdna_optimization.OptimizerState(
            observables={"trajectory": MockRef(value=jnp.array([1.0]))},
            component_state={"test_obj": {"opt_steps": 10, "call_count": 3}},
        )
        params = {"param": jnp.array(1.0)}
        opt.step(params=params, state=initial_state)

        assert received_state.get("opt_steps") == 10
        assert received_state.get("call_count") == 3

    def test_step_passes_objective_state_on_retry(
        self, basic_simulator, basic_optimizer
    ):
        call_records = []

        @chex.dataclass(frozen=True, kw_only=True)
        class StatefulObjective(jdna_objective.Objective):
            @override
            def compute(self, observables, opt_params, **state):
                call_records.append(dict(state))
                call_count = state.get("call_count", 0)

                if call_count == 0:
                    # First call: not ready, return state that should be passed back
                    return jdna_objective.ObjectiveOutput(
                        is_ready=False,
                        grads=None,
                        observables={},
                        state={
                            "call_count": 1,
                            "accumulated_data": [1, 2, 3],
                            "important_state": "first_run",
                        },
                        needs_update=("trajectory",),
                    )
                # Second call: should receive the state from first call
                return jdna_objective.ObjectiveOutput(
                    is_ready=True,
                    grads={k: v * 0.1 for k, v in opt_params.items()},
                    observables={"total": jnp.array(1.0)},
                    state={
                        "call_count": call_count + 1,
                        "accumulated_data": [*state.get("accumulated_data", []), 4],
                        "important_state": "second_run",
                    },
                )

        objective = StatefulObjective(
            name="stateful_obj",
            required_observables=("trajectory",),
            grad_or_loss_fn=lambda x: x,
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[objective],
            simulators=[basic_simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=basic_optimizer,
        )

        # Start with observables so objective gets called immediately
        initial_state = jdna_optimization.OptimizerState(
            observables={"trajectory": MockRef(value=jnp.array([1.0]))},
        )
        params = {"param": jnp.array(1.0)}
        output = opt.step(params=params, state=initial_state)

        # Verify we had at least 2 calls
        assert len(call_records) >= 2

        # First call should have no/empty state
        assert call_records[0].get("call_count", 0) == 0

        # Second call MUST receive the state from the first call's output
        assert call_records[1].get("call_count") == 1
        assert call_records[1].get("accumulated_data") == [1, 2, 3]
        assert call_records[1].get("important_state") == "first_run"

        # Final state should have the last state
        assert output.state.component_state["stateful_obj"]["call_count"] == 2
        assert output.state.component_state["stateful_obj"]["accumulated_data"] == [1, 2, 3, 4]

    def test_step_returns_output_observables(
        self, basic_simulator, basic_optimizer
    ):
        objective = MockObjective(
            name="test_obj",
            required_observables=("trajectory",),
            grad_or_loss_fn=lambda x: x,
            output_observables={
                "total": jnp.array(6.0),
                "mean": jnp.array(2.0),
                "custom_metric": jnp.array(42.0),
            },
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[objective],
            simulators=[basic_simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=basic_optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = opt.step(params=params)

        obj_observables = output.observables.get("test_obj", {})

        assert "total" in obj_observables
        assert "mean" in obj_observables
        assert "custom_metric" in obj_observables
        assert float(obj_observables["total"]) == 6.0
        assert float(obj_observables["mean"]) == 2.0
        assert float(obj_observables["custom_metric"]) == 42.0

    def test_step_initializes_optimizer_state(
        self, basic_objective, basic_simulator
    ):
        opt = jdna_optimization.RayOptimizer(
            objectives=[basic_objective],
            simulators=[basic_simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=optax.adam(0.01),
        )

        params = {"param": jnp.array(1.0)}
        output = opt.step(params=params)

        assert output.state.optimizer_state is not None

    def test_step_preserves_optimizer_state_across_steps(
        self, basic_objective, basic_simulator
    ):
        opt = jdna_optimization.RayOptimizer(
            objectives=[basic_objective],
            simulators=[basic_simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=optax.adam(0.01),
        )

        params = {"param": jnp.array(1.0)}
        output1 = opt.step(params=params)
        assert output1.state.optimizer_state is not None

        output2 = opt.step(params=output1.opt_params, state=output1.state)
        assert output2.state.optimizer_state is not None

    def test_step_stores_objective_state_in_optimizer_state(
        self, basic_simulator, basic_optimizer
    ):
        objective = MockObjective(
            name="test_obj",
            required_observables=("trajectory",),
            grad_or_loss_fn=lambda x: x,
            output_state={"custom_obj_data": "abc", "obj_call_count": 1},
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[objective],
            simulators=[basic_simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=basic_optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = opt.step(params=params)

        assert "test_obj" in output.state.component_state
        assert output.state.component_state["test_obj"]["obj_call_count"] == 1
        assert output.state.component_state["test_obj"]["custom_obj_data"] == "abc"

    def test_step_stores_simulator_state_in_optimizer_state(
        self, basic_objective, basic_optimizer
    ):
        simulator = MockSimulator(
            name="test_sim",
            output_state={"sim_call_count": 1, "custom_sim_data": "xyz"},
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[basic_objective],
            simulators=[simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=basic_optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = opt.step(params=params)

        assert "test_sim" in output.state.component_state
        assert output.state.component_state["test_sim"]["sim_call_count"] == 1
        assert output.state.component_state["test_sim"]["custom_sim_data"] == "xyz"

    def test_step_with_multiple_objectives_and_simulators(
        self, basic_optimizer
    ):
        objective1 = MockObjective(
            name="obj_1",
            required_observables=("trajectory.sim_1",),
            grad_or_loss_fn=lambda x: x,
        )
        objective2 = MockObjective(
            name="obj_2",
            required_observables=("trajectory.sim_2",),
            grad_or_loss_fn=lambda x: x,
        )
        simulator1 = MockSimulator(
            name="sim_1",
            return_observables=[jnp.array([1.0])],
            exposed_observables=["trajectory.sim_1"],
        )
        simulator2 = MockSimulator(
            name="sim_2",
            return_observables=[jnp.array([2.0])],
            exposed_observables=["trajectory.sim_2"],
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[objective1, objective2],
            simulators=[simulator1, simulator2],
            aggregate_grad_fn=lambda grads: {k: sum(g[k] for g in grads) / len(grads) for k in grads[0]},
            optimizer=basic_optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = opt.step(params=params)

        assert output.grads is not None
        assert output.opt_params is not None
        # Both objectives should have contributed to output
        assert "total" in output.observables["obj_1"]
        assert "total" in output.observables["obj_2"]

    def test_step_with_multi_observable_simulator_and_objective(
        self, basic_optimizer
    ):
        @chex.dataclass(frozen=True, kw_only=True)
        class MultiInputObjective(jdna_objective.Objective):
            @override
            def compute(self, observables, opt_params, **_state):
                sums = {f"{k}_sum": v.sum() for k, v in observables.items()}
                combined = sum([i.sum() for i in observables.values()])
                return jdna_objective.ObjectiveOutput(
                    is_ready=True,
                    grads={k: v * 0.1 for k, v in opt_params.items()},
                    observables={
                        "combined_total": combined,
                        **sums,
                    },
                    state={"call_count": 1},
                )

        simulator = MockSimulator(
            name="multi_sim",
            return_observables=[
                jnp.array([1.0, 2.0, 3.0]),  # trajectory
                jnp.array([10.0, 20.0]),     # energy
                jnp.array([0.5]),            # temperature
            ],
            exposed_observables=[
                "trajectory.multi_sim",
                "energy.multi_sim",
                "temperature.multi_sim",
            ],
        )
        objective = MultiInputObjective(
            name="multi_obj",
            required_observables=("trajectory.multi_sim", "energy.multi_sim", "temperature.multi_sim"),
            grad_or_loss_fn=lambda x: x,
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[objective],
            simulators=[simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=basic_optimizer,
        )

        params = {"param": jnp.array(1.0)}
        output = opt.step(params=params)

        assert output.grads is not None
        assert output.opt_params is not None
        # Verify all observables were properly passed and computed
        assert output.observables["multi_obj"]["combined_total"] == 36.5
        for obs, res in zip(simulator.exposes(), [6.0, 30.0, 0.5], strict=True):
            assert output.observables["multi_obj"][f"{obs}_sum"] == res


    def test_step_handles_objective_not_ready(
        self, basic_simulator, basic_optimizer
    ):
        call_count = [0]

        @chex.dataclass(frozen=True, kw_only=True)
        class ConditionalReadyObjective(jdna_objective.Objective):
            @override
            def compute(self, observables, opt_params, **state):
                call_count[0] += 1
                if call_count[0] == 1:
                    return jdna_objective.ObjectiveOutput(
                        is_ready=False,
                        grads=None,
                        observables={},
                        state={"call_count": call_count[0]},
                        needs_update=("trajectory",),
                    )
                return jdna_objective.ObjectiveOutput(
                    is_ready=True,
                    grads={k: v * 0.1 for k, v in opt_params.items()},
                    observables={"total": jnp.array(1.0)},
                    state={"call_count": call_count[0]},
                )

        objective = ConditionalReadyObjective(
            name="test_obj",
            required_observables=("trajectory",),
            grad_or_loss_fn=lambda x: x,
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[objective],
            simulators=[basic_simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=basic_optimizer,
        )

        initial_state = jdna_optimization.OptimizerState(
            observables={"trajectory": MockRef(value=jnp.array([0.5]))},
        )
        params = {"param": jnp.array(1.0)}
        output = opt.step(params=params, state=initial_state)

        assert output.grads is not None
        assert call_count[0] >= 2  # Should have been called at least twice

    def test_step_updates_params_using_optimizer(
        self, basic_simulator
    ):

        @chex.dataclass(frozen=True, kw_only=True)
        class FixedGradObjective(jdna_objective.Objective):
            @override
            def compute(self, observables, opt_params, **_state):
                return jdna_objective.ObjectiveOutput(
                    is_ready=True,
                    grads={"param": jnp.array(1.0)},  # Fixed gradient of 1.0
                    observables={"total": jnp.array(1.0)},
                    state={"call_count": 1},
                )

        objective = FixedGradObjective(
            name="test_obj",
            required_observables=("trajectory",),
            grad_or_loss_fn=lambda x: x,
        )

        learning_rate = 0.1
        opt = jdna_optimization.RayOptimizer(
            objectives=[objective],
            simulators=[basic_simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=optax.sgd(learning_rate),
        )

        initial_param = jnp.array(5.0)
        params = {"param": initial_param}
        output = opt.step(params=params)

        # With SGD: new_param = old_param - learning_rate * gradient
        expected_param = initial_param - learning_rate * 1.0
        assert jnp.allclose(output.opt_params["param"], expected_param)

    def test_step_raises_on_unresolvable_objective(self, basic_simulator, basic_optimizer):

        @chex.dataclass(frozen=True, kw_only=True)
        class NeverReadyObjective(jdna_objective.Objective):
            @override
            def compute(self, observables, opt_params, **_state):
                return jdna_objective.ObjectiveOutput(
                    is_ready=False,
                    needs_update=("trajectory",),
                    grads={},
                    observables={},
                    state={},
                )

        objective = NeverReadyObjective(
            name="never_ready",
            required_observables=("trajectory",),
            grad_or_loss_fn=lambda x: x,
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[objective],
            simulators=[basic_simulator],
            aggregate_grad_fn=lambda grads: grads[0],
            optimizer=basic_optimizer,
        )

        params = {"param": jnp.array(1.0)}

        with pytest.raises(RuntimeError, match="could not be resolved after multiple attempts"):
            opt.step(params=params)
