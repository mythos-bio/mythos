"""Tests for mythos.optimization.objective"""

import pathlib
import typing
from collections.abc import Callable

import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

import mythos.optimization.objective as o
import mythos.simulators.io as jdna_sio
import mythos.utils.types as jdna_types
from mythos.energy.base import EnergyFunction

file_location = pathlib.Path(__file__).parent
data_dir = file_location / "data"


def mock_return_function(should_return: typing.Any) -> Callable:
    """Return a function that returns the given value."""

    def mock_function(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return should_return

    return mock_function


def make_mock_energy_fn(return_value = None) -> EnergyFunction:
    class MockEF:
        def __call__(self, n):
            return n.sum()

        def map(self, n):
            if return_value is not None:
                return return_value
            return jnp.array(n)

        def with_params(self, *_args, **_kwargs):
            return self

    return MockEF()

@pytest.fixture
def mock_energy_fn():
    return make_mock_energy_fn()


@pytest.mark.parametrize(
    ("required_observables", "logging_observables", "grad_or_loss_fn", "expected_missing"),
    [
        (None, ["c"], lambda x: x, "required_observables"),
        (["a"], None, lambda x: x, "logging_observables"),
        (["a"], ["c"], None, "grad_or_loss_fn"),
    ],
)
def test_objective_init_raises(
    required_observables: list[str],
    logging_observables: list[str],
    grad_or_loss_fn: typing.Callable[[tuple[str, ...]], jdna_types.Grads],
    expected_missing: str,
) -> None:
    """Test the __init__ function for Objective raises for missing required arg."""

    with pytest.raises(ValueError, match=o.ERR_MISSING_ARG.format(missing_arg=expected_missing)):
        o.Objective(
            name="test",
            required_observables=required_observables,
            logging_observables=logging_observables,
            grad_or_loss_fn=grad_or_loss_fn,
        )


def test_objective_needed_observables() -> None:
    """Test the needed_observables property of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=["a", "b"],
        logging_observables=["c"],
        grad_or_loss_fn=lambda x: x,
    )

    obj.update_one("b", 2.0)
    assert obj.needed_observables() == {"a"}


def test_objective_logging_observables() -> None:
    """Test the logging_observables property of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=["a", "b", "c"],
        logging_observables=["a", "b"],
        grad_or_loss_fn=lambda x: x,
    )
    obj.update(["a", "b", "c"], 1.0, 2.0, 3.0)
    # we are only logging two so we should only get those
    expected = {"a": 1.0, "b": 2.0}
    assert obj.logging_observables() == expected


@pytest.mark.parametrize(
    ("required_observables", "obtained_observables", "expected"),
    [
        (["a"], [("a"), 1.0], True),
        (["a"], [("b"), 1.0], False),
        (["a", "b"], [("a", "b"), 1.0, 2.0], True),
        (["a", "b"], [("a"), 1.0], False),
    ],
)
def test_objective_is_ready(
    required_observables: list[str],
    obtained_observables: list[tuple[str, float]],
    expected: bool,  # noqa: FBT001
) -> None:
    """Test the is_ready method of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=required_observables,
        logging_observables=[],
        grad_or_loss_fn=lambda x: x,
    )
    obj.update(*obtained_observables)

    assert obj.is_ready() == expected


@pytest.mark.parametrize(
    ("required_observables", "update_collection", "expected_needed"),
    [
        (["a", "b"], [("a"), {"test": 1.0}], {"b"}),
        (["a"], [("a", "b"), {"test": 1.0}, {"test2": 2.0}], set()),
    ],
)
def test_objective_update(
    required_observables: list[str],
    update_collection: list[tuple[str, str]],
    expected_needed: list[str],
) -> None:
    """Test the update method of Objective."""

    obj = o.Objective(
        name="test",
        required_observables=required_observables,
        logging_observables=[],
        grad_or_loss_fn=lambda x: x,
    )

    obj.update(*update_collection)
    assert obj.needed_observables() == expected_needed


def test_objective_calculate() -> None:
    """Test the calculate method of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=["a", "b", "c"],
        logging_observables=[],
        grad_or_loss_fn=mock_return_function((1.0, [("test", 0.0)])),
    )

    # simulate getting the observables
    obj.update(["a", "b", "c"], 1.0, 2.0, 3.0)

    # simulate the calculate
    result = obj.calculate()

    assert result == 1.0


def test_objective_post_step() -> None:
    obj = o.Objective(
        name="test",
        required_observables=["a", "b", "c"],
        logging_observables=[],
        grad_or_loss_fn=lambda x: x,
    )

    # simulate getting the observables
    obj.update_one("c", 3.0)

    # simulate the post step
    obj.post_step(opt_params={})

    assert obj.needed_observables() == {"a", "b", "c"}


@pytest.mark.parametrize(
    ("beta", "new_energies", "ref_energies", "expected_weights", "expected_neff"),
    [
        (1, np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1 / 3, 1 / 3, 1 / 3]), np.array(1.0, dtype=np.float64)),
    ],
)
def test_compute_weights_and_neff(
    beta: float,
    new_energies: np.ndarray,
    ref_energies: np.ndarray,
    expected_weights: np.ndarray,
    expected_neff: float,
) -> None:
    """Test the weights calculation in for a Difftre Objective."""
    weights, neff = o.compute_weights_and_neff(beta, new_energies, ref_energies)
    assert np.allclose(weights, expected_weights)
    assert np.allclose(neff, expected_neff)


@pytest.mark.parametrize(
    (
        "opt_params",
        "beta",
        "ref_states",
        "ref_energies",
        "expected_loss",
        "expected_measured_value",
    ),
    [
        (
            {},
            1.0,
            jdna_sio.SimulatorTrajectory(rigid_body=np.array([1, 2, 3])),
            jdna_sio.SimulatorTrajectory(rigid_body=np.array([1, 2, 3])),
            0.0,
            ("test", 1.0),
        ),
    ],
)
def test_compute_loss(
    opt_params: dict[str, float],
    beta: float,
    ref_states: np.ndarray,
    ref_energies: np.ndarray,
    expected_loss: float,
    expected_measured_value: tuple[str, float],
    mock_energy_fn,
) -> None:
    """Test the loss calculation in for a Difftre Objective."""
    expected_aux = (np.array(1.0), expected_measured_value, np.array([1, 2, 3]))
    loss_fn = mock_return_function((expected_loss, (expected_measured_value, {})))
    ref_energies = mock_energy_fn.map(ref_energies.rigid_body)

    loss, aux = o.compute_loss(opt_params, mock_energy_fn, beta, loss_fn, ref_states, ref_energies, observables=[])

    assert loss == expected_loss

    def eq(a, b) -> bool:
        if isinstance(a, np.ndarray | jnp.ndarray):
            assert np.allclose(a, b)
        elif isinstance(a, tuple):
            [eq(x, y) for x, y in zip(a, b, strict=True)]
        else:
            assert a == b

    for a, ea in zip(aux, expected_aux, strict=True):
        eq(a, ea)


@pytest.mark.parametrize(
    ("energy_fn", "opt_params", "beta", "n_equilibration_steps", "missing_arg"),
    [
        (None, {}, 1.0, 1, "energy_fn"),
        (lambda _: mock_return_function(np.array([1, 2, 3])), None, 1.0, 1, "opt_params"),
        (lambda _: mock_return_function(np.array([1, 2, 3])), {"a": 1}, None, 1, "beta"),
        (lambda _: mock_return_function(np.array([1, 2, 3])), {"a": 1}, 1.0, None, "n_equilibration_steps"),
    ],
)
def test_difftreobjective_init_raises(
    energy_fn: Callable[[jdna_types.Params], Callable[[np.ndarray], np.ndarray]],
    opt_params: jdna_types.Params,
    beta: float,
    n_equilibration_steps: int,
    missing_arg: str,
) -> None:
    required_observables = ["a"]
    logging_observables = ["c"]
    grad_or_loss_fn = lambda x: x

    with pytest.raises(ValueError, match=o.ERR_MISSING_ARG.format(missing_arg=missing_arg)):
        o.DiffTReObjective(
            name="test",
            required_observables=required_observables,
            logging_observables=logging_observables,
            grad_or_loss_fn=grad_or_loss_fn,
            energy_fn=energy_fn,
            opt_params=opt_params,
            beta=beta,
            n_equilibration_steps=n_equilibration_steps,
        )


def test_difftreobjective_calculate() -> None:
    """Test the calculate method of DifftreObjective."""
    obj = o.DiffTReObjective(
        name="test",
        required_observables=["test"],
        logging_observables=[],
        grad_or_loss_fn=mock_return_function((1.0, (("test", 1.0), {}))),
        energy_fn=make_mock_energy_fn(jnp.ones(100)),
        opt_params={"test": 1.0},
        beta=1.0,
        n_equilibration_steps=10,
    )

    # simulate getting the observables
    obj.update_one("test",
        jdna_sio.SimulatorTrajectory(
            rigid_body=jax_md.rigid_body.RigidBody(
                center=np.arange(110),
                orientation=jax_md.rigid_body.Quaternion(
                    vec=np.arange(440).reshape(110, 4),
                ),
            )
        ),
    )

    # simulate the calculate
    expected_grad = {"test": jnp.array(0.0)}
    actual_grad = obj.calculate()

    assert actual_grad == expected_grad


def test_difftreobjective_post_step(mock_energy_fn) -> None:
    """test thge post_step method of DiffTReObjective."""
    obj = o.DiffTReObjective(
        name="test",
        required_observables=["test"],
        logging_observables=[],
        grad_or_loss_fn=mock_return_function((1.0, 0.0)),
        energy_fn=mock_energy_fn,
        opt_params={"test": 1.0},
        beta=1.0,
        n_equilibration_steps=10,
    )

    obj.update(("test", "loss"), "some array data", 1.0)

    # run the post step
    new_params = {"test": 2.0}
    obj.post_step(opt_params=new_params)

    assert obj._obtained_observables == {"test": "some array data"}
    assert obj._opt_params == new_params
