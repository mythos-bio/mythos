"""Tests for optimization module."""

from dataclasses import field
from typing import Any

import chex
import numpy as np
import optax
import pytest
import ray

from mythos.optimization import optimization
from mythos.optimization.objective import Objective
from mythos.simulators.base import AsyncSimulation


@pytest.fixture(autouse=True)
def _ray_mocking(monkeypatch):
    """Fixture to mock ray.get and ray.wait in optimization tests."""

    def mock_get_fn(value: Any):
        return value

    def ray_wait_fn(refs: list[Any], **kwargs):
        return refs, []

    monkeypatch.setattr(ray, "get", mock_get_fn)
    monkeypatch.setattr(ray, "wait", ray_wait_fn)
    monkeypatch.setattr(optax, "apply_updates", lambda x, y: (x, y))


class MockRayObjective(Objective):
    def __init__(self, *args, calc_value=None, ready=False, required_observables=None, **kwargs):
        required_observables = required_observables or []
        super().__init__(
            *args,
            **kwargs,
            required_observables=required_observables,
            logging_observables=[],
            grad_or_loss_fn=lambda *_obs: (0, [("obs", 1)]),
        )
        self.calc_value = calc_value
        if ready:
            self._obtained_observables = {k: 1 for k in self._required_observables}

    def calculate_async(self):
        return self.calc_value


@chex.dataclass(eq=False)
class MockRaySimulator(AsyncSimulation):
    expose_values: list[str] = field(default_factory=list)
    run_value: Any | None = None

    def run(self, _params):
        return self.run_value

    def run_async(self, _params):
        return self.run_value

    def exposes(self):
        return self.expose_values


class MockOptimizer:
    def init(self, params):
        self.n_update_calls = 0
        return params

    def update(self, grads, opt_state, params):  # noqa: ARG002 -- This is just for testing
        self.n_update_calls += 1
        return {}, opt_state


@pytest.mark.parametrize(
    ("objectives", "simulators", "aggregate_grad_fn", "optimizer", "expected_err"),
    [
        ([], [MockRaySimulator()], lambda x: x, MockOptimizer(), optimization.ERR_MISSING_OBJECTIVES),
        (
            [MockRayObjective(name="test", ready=True)],
            [],
            lambda x: x,
            MockOptimizer(),
            optimization.ERR_MISSING_SIMULATORS,
        ),
        (
            [MockRayObjective(name="test", ready=True)],
            [MockRaySimulator()],
            None,
            MockOptimizer(),
            optimization.ERR_MISSING_AGG_GRAD_FN,
        ),
        (
            [MockRayObjective(name="test", ready=True)],
            [MockRaySimulator()],
            lambda x: x,
            None,
            optimization.ERR_MISSING_OPTIMIZER,
        ),
    ],
)
def test_optimization_post_init_raises(
    objectives,
    simulators,
    aggregate_grad_fn,
    optimizer,
    expected_err,
):
    with pytest.raises(ValueError, match=expected_err):
        optimization.RayMultiOptimizer(
            objectives=objectives, simulators=simulators, aggregate_grad_fn=aggregate_grad_fn, optimizer=optimizer
        )


def test_optimzation_step():
    """Test that the optimization step."""

    opt = optimization.RayMultiOptimizer(
        objectives=[
            MockRayObjective(name="test", ready=True, calc_value=1, required_observables=["q_1"]),
            MockRayObjective(name="test", ready=False, calc_value=2, required_observables=["q_2"]),
        ],
        simulators=[
            MockRaySimulator(name="test", run_value="test-1", expose_values=["q_1"]),
            MockRaySimulator(name="test", run_value="test-2", expose_values=["q_2"]),
        ],
        aggregate_grad_fn=np.mean,
        optimizer=MockOptimizer(),
    )

    opt_state, params, _ = opt.step(params={"test": 1})
    assert opt_state is not None
    assert params == ({"test": 1}, {})


def test_optimization_post_step():
    """Test that the optimizer state is updated after a step."""
    opt = optimization.RayMultiOptimizer(
        objectives=[MockRayObjective(name="test", ready=True)],
        simulators=[MockRaySimulator()],
        aggregate_grad_fn=lambda x: x,
        optimizer=MockOptimizer(),
        optimizer_state="old",
    )

    new_state = "new"
    opt = opt.post_step(optimizer_state=new_state, opt_params={})
    assert opt.optimizer_state == new_state


def test_optimization_fails_for_redundant_observables():
    """Test that the optimization fails if two objectives require the same observable."""
    with pytest.raises(ValueError, match="expose the same observable"):
        optimization.RayMultiOptimizer(
            objectives=[MockRayObjective(name="test2", ready=True, required_observables=["q_1"])],
            simulators=[MockRaySimulator(expose_values=["q_1", "q_2"]), MockRaySimulator(expose_values=["q_2", "q_3"])],
            aggregate_grad_fn=lambda x: x,
            optimizer=MockOptimizer(),
        )


@pytest.mark.parametrize("obj_ready", [True, False])
def test_simple_optimizer(obj_ready):
    """Test that the SimpleOptimizer can be created."""
    opt = optimization.SimpleOptimizer(
        objective=MockRayObjective(name="test", ready=True, calc_value=1, required_observables=["q_1"]),
        simulator=MockRaySimulator(name="test", run_value="test-2", expose_values=["q_1"]),
        optimizer=MockOptimizer(),
    )

    opt_state, params, _ = opt.step(params={"test": 1})
    assert opt_state is not None
    assert params == ({"test": 1}, {})
    new_opt = opt.post_step(optimizer_state=opt_state, opt_params=params)
    assert isinstance(new_opt, optimization.SimpleOptimizer)
