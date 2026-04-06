"""Integration tests that use a real (local) Ray session."""

from dataclasses import field

import chex
import jax.numpy as jnp
import optax
import pytest
import ray
from typing_extensions import override

import mythos.optimization.objective as jdna_objective
import mythos.optimization.optimization as jdna_optimization
from mythos.simulators.base import Simulator, SimulatorOutput


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    """Start a minimal local Ray session for the module and shut it down after."""
    ray.init(num_cpus=2, log_to_driver=False)
    yield
    ray.shutdown()


# ---------------------------------------------------------------------------
# Minimal mock components (must be picklable for Ray serialization)
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True, kw_only=True)
class RayMockSimulator(Simulator):
    name: str = "ray_sim"
    return_observables: list[jnp.ndarray] = field(default_factory=lambda: [jnp.array([1.0, 2.0])])
    exposed_observables: list[str] = field(default_factory=lambda: ["trajectory"])

    @override
    def run(self, *_args, opt_params, **state) -> SimulatorOutput:
        return SimulatorOutput(
            observables=self.return_observables,
            state={"call_count": state.get("call_count", 0) + 1},
        )

    def exposes(self) -> list[str]:
        return self.exposed_observables


@chex.dataclass(frozen=True, kw_only=True)
class RayMockObjective(jdna_objective.Objective):
    @override
    def calculate(self, observables, opt_params, **state):
        obs = next(iter(observables.values()))
        grads = {k: v * 0.1 for k, v in opt_params.items()}
        return jdna_objective.ObjectiveOutput(
            is_ready=True,
            state={"call_count": state.get("call_count", 0) + 1},
            observables={"total": obs.sum()},
            grads=grads,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRayIntegration:
    """Tests that exercise the real Ray remote machinery."""

    def test_single_step_through_ray(self):
        """A single optimisation step runs end-to-end via Ray remotes."""
        simulator = RayMockSimulator()
        objective = RayMockObjective(
            name="ray_obj",
            required_observables=tuple(simulator.exposes()),
            grad_or_loss_fn=lambda x: x,
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[objective],
            simulators=[simulator],
            aggregate_grad_fn=lambda grads: grads[0] if grads else {},
            optimizer=optax.sgd(0.01),
        )

        params = {"p": jnp.array(1.0)}
        output = opt.step(params=params)

        assert isinstance(output, jdna_optimization.OptimizerOutput)
        assert output.opt_params is not None
        assert output.grads is not None
        # SGD: new = old - lr * grad;  grad = 0.1 * 1.0 = 0.1
        assert jnp.allclose(output.opt_params["p"], 1.0 - 0.01 * 0.1)
        # observable recorded
        assert "ray_obj" in output.observables
        assert float(output.observables["ray_obj"]["total"]) == 3.0  # 1+2

    def test_multi_step_state_persists_through_ray(self):
        """State flows correctly across two consecutive Ray-backed steps."""
        simulator = RayMockSimulator()
        objective = RayMockObjective(
            name="ray_obj",
            required_observables=tuple(simulator.exposes()),
            grad_or_loss_fn=lambda x: x,
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[objective],
            simulators=[simulator],
            aggregate_grad_fn=lambda grads: grads[0] if grads else {},
            optimizer=optax.sgd(0.01),
        )

        params = {"p": jnp.array(5.0)}
        out1 = opt.step(params=params)
        out2 = opt.step(params=out1.opt_params, state=out1.state)

        # Params should have been updated twice
        assert not jnp.allclose(out1.opt_params["p"], out2.opt_params["p"])
        # Optimizer state preserved
        assert out2.state.optimizer_state is not None
        # Component states carried forward
        assert out2.state.component_state["ray_obj"]["call_count"] >= 1
        assert out2.state.component_state["ray_sim"]["call_count"] >= 1

    def test_multiple_simulators_and_objectives_through_ray(self):
        """Two simulator / objective pairs coordinate correctly via Ray."""
        sim1 = RayMockSimulator(
            name="sim_a",
            return_observables=[jnp.array([10.0])],
            exposed_observables=["obs_a"],
        )
        sim2 = RayMockSimulator(
            name="sim_b",
            return_observables=[jnp.array([20.0])],
            exposed_observables=["obs_b"],
        )
        obj1 = RayMockObjective(
            name="obj_a",
            required_observables=("obs_a",),
            grad_or_loss_fn=lambda x: x,
        )
        obj2 = RayMockObjective(
            name="obj_b",
            required_observables=("obs_b",),
            grad_or_loss_fn=lambda x: x,
        )

        opt = jdna_optimization.RayOptimizer(
            objectives=[obj1, obj2],
            simulators=[sim1, sim2],
            aggregate_grad_fn=lambda grads: {k: sum(g[k] for g in grads) / len(grads) for k in grads[0]},
            optimizer=optax.sgd(0.01),
        )

        params = {"p": jnp.array(1.0)}
        output = opt.step(params=params)

        assert "obj_a" in output.observables
        assert "obj_b" in output.observables
        assert float(output.observables["obj_a"]["total"]) == 10.0
        assert float(output.observables["obj_b"]["total"]) == 20.0
