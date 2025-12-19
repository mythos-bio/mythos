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
            return jnp.array(n.center)

        def with_params(self, *_args, **_kwargs):
            return self

    return MockEF()

@pytest.fixture
def mock_energy_fn():
    return make_mock_energy_fn()


@pytest.mark.parametrize(
    ("required_observables", "logging_observables", "grad_or_loss_fn", "expected_missing"),
    [
        (None, ("c",), lambda x: x, "required_observables"),
        (("a",), ("c",), None, "grad_or_loss_fn"),
    ],
)
def test_objective_init_raises(
    required_observables: tuple[str, ...] | None,
    logging_observables: tuple[str, ...],
    grad_or_loss_fn: typing.Callable[[tuple[str, ...]], jdna_types.Grads] | None,
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


def test_objective_required_observables() -> None:
    """Test the required_observables property of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=("a",),
        logging_observables=("c",),
        grad_or_loss_fn=lambda x: (x, []),
    )

    assert obj.required_observables == ("a",)


def test_objective_logging_observables() -> None:
    """Test the get_logging_observables method of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=("a", "b", "c"),
        logging_observables=("a", "b"),
        grad_or_loss_fn=lambda x: (x, []),
    )

    observables = {
        "a": 1.0,
        "b": 2.0,
        "c": 3.0,
    }

    # we are only logging two so we should only get those
    expected = [
        ("a", 1.0),
        ("b", 2.0),
    ]
    assert obj.get_logging_observables(observables) == expected


@pytest.mark.parametrize(
    ("required_observables", "observables", "expected"),
    [
        (("a",), {"a": 1.0}, True),
        (("a",), {"b": 1.0}, False),
        (("a", "b"), {"a": 1.0, "b": 2.0}, True),
        (("a", "b"), {"a": 1.0}, False),
    ],
)
def test_objective_compute_is_ready(
    required_observables: tuple[str, ...],
    observables: dict[str, float],
    expected: bool,  # noqa: FBT001
) -> None:
    """Test the compute method's is_ready output of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=required_observables,
        logging_observables=(),
        grad_or_loss_fn=lambda *args: (args, []),
    )

    output = obj.compute(observables)
    assert output.is_ready == expected


def test_objective_compute() -> None:
    """Test the compute method of Objective."""
    obj = o.Objective(
        name="test",
        required_observables=("a", "b", "c"),
        logging_observables=(),
        grad_or_loss_fn=mock_return_function((1.0, [("test", 0.0)])),
    )

    observables = {
        "a": 1.0,
        "b": 2.0,
        "c": 3.0,
    }

    output = obj.compute(observables)

    assert output.is_ready
    assert output.grads == 1.0
    assert output.needs_update == ()


def test_objective_compute_returns_needs_update() -> None:
    """Test that compute returns needs_update when observables are missing."""
    obj = o.Objective(
        name="test",
        required_observables=("a", "b", "c"),
        logging_observables=(),
        grad_or_loss_fn=lambda x: (x, []),
    )

    observables = {"a": 1.0}

    output = obj.compute(observables)

    assert not output.is_ready
    assert set(output.needs_update) == {"b", "c"}


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
            jdna_sio.SimulatorTrajectory(
                rigid_body=jax_md.rigid_body.RigidBody(
                    center=np.array([1, 2, 3]),
                    orientation=np.array([1, 0, 0, 0]),
                ),
            ),
            jdna_sio.SimulatorTrajectory(
                rigid_body=jax_md.rigid_body.RigidBody(
                    center=np.array([1, 2, 3]),
                    orientation=np.array([1, 0, 0, 0]),
                ),
            ),
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
    ("energy_fn", "beta", "n_equilibration_steps", "missing_arg"),
    [
        (None, 1.0, 1, "energy_fn"),
        (lambda _: mock_return_function(np.array([1, 2, 3])), None, 1, "beta"),
        (lambda _: mock_return_function(np.array([1, 2, 3])), 1.0, None, "n_equilibration_steps"),
    ],
)
def test_difftreobjective_init_raises(
    energy_fn: Callable[[jdna_types.Params], Callable[[np.ndarray], np.ndarray]],
    beta: float,
    n_equilibration_steps: int,
    missing_arg: str,
) -> None:
    required_observables = ("a",)
    logging_observables = ("c",)
    grad_or_loss_fn = lambda x: x

    with pytest.raises(ValueError, match=o.ERR_MISSING_ARG.format(missing_arg=missing_arg)):
        o.DiffTReObjective(
            name="test",
            required_observables=required_observables,
            logging_observables=logging_observables,
            grad_or_loss_fn=grad_or_loss_fn,
            energy_fn=energy_fn,
            beta=beta,
            n_equilibration_steps=n_equilibration_steps,
        )


def test_difftreobjective_compute_raises_without_opt_params() -> None:
    """Test that DiffTReObjective.compute raises when opt_params is not provided."""
    obj = o.DiffTReObjective(
        name="test",
        required_observables=("test",),
        logging_observables=(),
        grad_or_loss_fn=mock_return_function((1.0, (("test", 1.0), {}))),
        energy_fn=make_mock_energy_fn(jnp.ones(100)),
        beta=1.0,
        n_equilibration_steps=10,
    )

    observables = {"test": 1.0}
    state = {"opt_steps": 1}

    with pytest.raises(TypeError, match="opt_params"):
        obj.compute(observables, **state)  # opt_params not provided


def test_difftreobjective_compute() -> None:
    """Test the compute method of DiffTReObjective."""
    obj = o.DiffTReObjective(
        name="test",
        required_observables=("test",),
        logging_observables=(),
        grad_or_loss_fn=mock_return_function((1.0, (("test", 1.0), {}))),
        energy_fn=make_mock_energy_fn(jnp.ones(100)),
        beta=1.0,
        n_equilibration_steps=10,
    )

    # Create a trajectory observable
    trajectory = jdna_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=np.arange(110),
            orientation=jax_md.rigid_body.Quaternion(
                vec=np.arange(440).reshape(110, 4),
            ),
        )
    )

    observables = {"test": trajectory}
    state = {"opt_steps": 1}
    opt_params = {"test": 1.0}

    output = obj.compute(observables, **state, opt_params=opt_params)

    expected_grad = {"test": jnp.array(0.0)}
    assert output.is_ready
    assert output.grads == expected_grad
    assert output.needs_update == ()


def test_difftreobjective_compute_returns_needs_update_when_missing() -> None:
    """Test that DiffTReObjective returns needs_update when observables are missing."""
    obj = o.DiffTReObjective(
        name="test",
        required_observables=("test",),
        logging_observables=(),
        grad_or_loss_fn=mock_return_function((1.0, (("test", 1.0), {}))),
        energy_fn=make_mock_energy_fn(jnp.ones(100)),
        beta=1.0,
        n_equilibration_steps=10,
    )

    # Empty observables - needs trajectory
    observables = {}
    state = {"opt_steps": 1}
    opt_params = {"test": 1.0}

    output = obj.compute(observables, **state, opt_params=opt_params)

    assert not output.is_ready
    assert "test" in output.needs_update


def test_difftreobjective_state_preserved() -> None:
    """Test that DiffTReObjective preserves reference states in state."""
    obj = o.DiffTReObjective(
        name="test",
        required_observables=("test",),
        logging_observables=(),
        grad_or_loss_fn=mock_return_function((1.0, (("measured", 1.0), {}))),
        energy_fn=make_mock_energy_fn(jnp.ones(100)),
        beta=1.0,
        n_equilibration_steps=10,
    )

    # Create a trajectory observable
    trajectory = jdna_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=np.arange(110),
            orientation=jax_md.rigid_body.Quaternion(
                vec=np.arange(440).reshape(110, 4),
            ),
        )
    )

    observables = {"test": trajectory}
    state = {"opt_steps": 1}
    opt_params = {"test": 1.0}

    output = obj.compute(observables, **state, opt_params=opt_params)

    # Output state should contain reference_states and reference_energies
    assert output.is_ready
    assert "reference_opt_params" in output.state
    assert output.state["reference_opt_params"] is not None
