# ruff: noqa: N802,FBT001,FBT002 - Ignore the BaseEnergyFunction/ComposableEnergyFunction names in functions and boolean arg rules
"""Tests for mythos.energy.base"""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import chex
import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

from mythos.energy import base
from mythos.energy.configuration import BaseConfiguration

NOT_IMPLEMENTED_ERR = re.compile("unsupported operand type\(s\) for")  # noqa: W605 - Ignore the regex warning


def _make_base_energy_function(
    with_displacement: bool = False,
) -> base.BaseEnergyFunction | tuple[base.BaseEnergyFunction, jax_md.space.DisplacementFn]:
    """Helper function to create a BaseEnergyFunction."""

    @chex.dataclass(frozen=True)
    class MockBaseEF(base.BaseEnergyFunction):
        params: None = None

        def compute_energy(self, _nucleotide):
            return None

    displacement_fn, _ = jax_md.space.free()
    top = MagicMock()
    be = MockBaseEF(displacement_fn=displacement_fn, transform_fn=None, topology=top)
    vals = be
    if with_displacement:
        vals = (be, displacement_fn)
    return vals


def test_BaseEnergyFunction_displacement_mapped() -> None:
    """Tests that the behavior of the displacement function is consistent."""
    a = np.array(
        [
            [1, 1],
            [0, 0],
        ]
    )
    b = np.array([[2, 1], [2, 0]])
    be, displacement_fn = _make_base_energy_function(with_displacement=True)
    np.testing.assert_allclose(be.displacement_mapped(a, b), jax_md.space.map_bond(displacement_fn)(a, b))


def test_BaseEnergyFunction_add() -> None:
    """Test the __add__ function for BaseEnergyFunction."""
    be = _make_base_energy_function()
    actual = be + be
    assert all(isinstance(e, base.BaseEnergyFunction) for e in actual.energy_fns)
    assert actual.weights is None


def test_BaseEnergyFunction_add_raises() -> None:
    """Test the __add__ function for BaseEnergyFunction with invalid args."""
    be = _make_base_energy_function()
    with pytest.raises(TypeError, match=NOT_IMPLEMENTED_ERR):
        be + 3


def test_BaseEnergyFunction_mul() -> None:
    """Test the __mul__ function for BaseEnergyFunction."""
    coef = 2
    be = _make_base_energy_function()
    actual = be * coef

    assert len(actual.energy_fns) == 1
    assert all(isinstance(e, base.BaseEnergyFunction) for e in actual.energy_fns)
    assert len(actual.weights) == 1
    assert actual.weights[0] == coef


def test_BaseEnergyFunction_mul_raises() -> None:
    """Test the __add__ function for BaseEnergyFunction with invalid args."""
    be = _make_base_energy_function()
    with pytest.raises(TypeError, match=NOT_IMPLEMENTED_ERR):
        be * be


def test_ComposedEnergyFunction_init() -> None:
    """Test the initialization params of ComposedEnergyFunction"""
    be = _make_base_energy_function()
    cef = base.ComposedEnergyFunction(energy_fns=[be])

    assert cef.energy_fns == [be]
    assert cef.weights is None


def test_ComposedEnergyFunction_init_raises() -> None:
    """Test the invalid initialization params of ComposedEnergyFunction"""
    be = _make_base_energy_function()
    expected_err = re.escape(base.ERR_COMPOSED_ENERGY_FN_TYPE_ENERGY_FNS)
    with pytest.raises(TypeError, match=expected_err):
        base.ComposedEnergyFunction(energy_fns=[be, 3])


def test_ComposedEnergyFunction_init_raises_lengths() -> None:
    """Test the __call__ function for ComposedEnergyFunction with invalid args."""
    be = _make_base_energy_function()
    with pytest.raises(ValueError, match=re.escape(base.ERR_COMPOSED_ENERGY_FN_LEN_MISMATCH)):
        base.ComposedEnergyFunction(energy_fns=[be], weights=np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    ("init_weights", "expected_weights"),
    [
        (None, None),
        (np.array([1.0]), np.array([1.0, 1.0])),
        (np.array([3.0]), np.array([3.0, 3.0])),
    ],
)
def test_ComposedEnergyFunction_add_energy_function(
    init_weights: np.ndarray | None,
    expected_weights: np.ndarray | None,
) -> None:
    """Test the add_energy_function method of ComposedEnergyFunction."""

    be = _make_base_energy_function()
    cef = base.ComposedEnergyFunction(energy_fns=[be], weights=init_weights)
    cef = cef.add_energy_fn(be, 1.0 if init_weights is None else init_weights[0])

    assert len(cef.energy_fns) == 2
    if init_weights is None:
        assert cef.weights is None
    else:
        np.testing.assert_allclose(cef.weights, expected_weights)


@pytest.mark.parametrize(
    ("init_weights", "expected_weights"),
    [
        (None, None),
        (np.array([1.0]), np.array([1.0, 1.0])),
        (np.array([3.0]), np.array([3.0, 3.0])),
    ],
)
def test_ComposedEnergyFunction_add_composable_energy_function(
    init_weights: np.ndarray | None,
    expected_weights: np.ndarray | None,
) -> None:
    """Test the add_composable_energy_function method of ComposedEnergyFunction."""

    be = _make_base_energy_function()
    cef = base.ComposedEnergyFunction(
        energy_fns=[be],
        weights=init_weights,
    ).add_composable_energy_fn(
        base.ComposedEnergyFunction(
            energy_fns=[be],
            weights=init_weights,
        )
    )

    assert len(cef.energy_fns) == 2
    if init_weights is None:
        assert cef.weights is None
    else:
        np.testing.assert_allclose(cef.weights, expected_weights)


@pytest.mark.parametrize(
    ("init_weights", "other", "expected_weights"),
    [
        (None, None, None),  # raises
        (None, _make_base_energy_function(), None),
        (np.array([1.0]), _make_base_energy_function(), np.array([1.0, 1.0])),
        (np.array([3.0]), _make_base_energy_function(), np.array([3.0, 1.0])),
        (None, base.ComposedEnergyFunction(energy_fns=[_make_base_energy_function()]), None),
        (np.array([1.0]), base.ComposedEnergyFunction(energy_fns=[_make_base_energy_function()]), np.array([1.0, 1.0])),
        (np.array([3.0]), base.ComposedEnergyFunction(energy_fns=[_make_base_energy_function()]), np.array([3.0, 1.0])),
    ],
)
def test_ComposedEnergyFunction_add(
    init_weights: np.ndarray | None,
    other: base.BaseEnergyFunction | base.ComposedEnergyFunction | None,
    expected_weights: np.ndarray | None,
) -> None:
    """Test the __add__ function for ComposedEnergyFunction."""
    be = _make_base_energy_function()
    cef = base.ComposedEnergyFunction(energy_fns=[be], weights=init_weights)

    if other is None:
        with pytest.raises(TypeError, match=NOT_IMPLEMENTED_ERR):
            cef + other
    else:
        cef = cef + other

        assert len(cef.energy_fns) == 2
        if init_weights is None:
            assert cef.weights is None
        else:
            np.testing.assert_allclose(cef.weights, expected_weights)


@pytest.mark.parametrize(
    ("init_weights", "other", "expected_weights"),
    [
        (None, None, None),  # raises
        (None, _make_base_energy_function(), None),
        (np.array([1.0]), _make_base_energy_function(), np.array([1.0, 1.0])),
        (np.array([3.0]), _make_base_energy_function(), np.array([3.0, 1.0])),
        (None, base.ComposedEnergyFunction(energy_fns=[_make_base_energy_function()]), None),
        (np.array([1.0]), base.ComposedEnergyFunction(energy_fns=[_make_base_energy_function()]), np.array([1.0, 1.0])),
        (np.array([3.0]), base.ComposedEnergyFunction(energy_fns=[_make_base_energy_function()]), np.array([1.0, 3.0])),
    ],
)
def test_ComposedEnergyFunction_radd(
    init_weights: np.ndarray | None,
    other: base.BaseEnergyFunction | base.ComposedEnergyFunction | None,
    expected_weights: np.ndarray | None,
) -> None:
    """Test the __add__ function for ComposedEnergyFunction."""
    be = _make_base_energy_function()
    cef = base.ComposedEnergyFunction(energy_fns=[be], weights=init_weights)

    if other is None:
        with pytest.raises(TypeError, match=NOT_IMPLEMENTED_ERR):
            other + cef
    else:
        cef = other + cef

        assert len(cef.energy_fns) == 2
        if init_weights is None:
            assert cef.weights is None
        else:
            np.testing.assert_allclose(cef.weights, expected_weights)


@chex.dataclass(frozen=True)
class MockEnergyFunction(base.BaseEnergyFunction):
    params: dict = None

    def compute_energy(self, nucleotide) -> float:
        return nucleotide.center.sum()


@pytest.mark.parametrize(
    ("rigid_body_transform_fn", "expected"),
    [
        (None, 4.0),
        (lambda x: jax_md.rigid_body.RigidBody(center=x.center * 2, orientation=x.orientation), 8.0),
    ],
)
def test_ComposedEnergyFunction_call(
    rigid_body_transform_fn: Callable | None,
    expected: float,
) -> None:
    """Test the __call__ function for ComposedEnergyFunction."""

    displacement_fn, _ = jax_md.space.free()
    be = MockEnergyFunction(displacement_fn=displacement_fn, transform_fn=rigid_body_transform_fn, topology=MagicMock())
    cef = base.ComposedEnergyFunction(energy_fns=[be])

    body = jax_md.rigid_body.RigidBody(
        center=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
        orientation=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
    )

    assert cef(body) == expected


@pytest.fixture
def list_of_energy_functions() -> list[base.BaseEnergyFunction]:
    """Fixture to create a list of BaseEnergyFunction instances."""

    @chex.dataclass(frozen=True)
    class MockParams1(BaseConfiguration):
        param1: float = 1.0
        param_shared: float = 3.0
        params_to_optimize: tuple[str] = ("*",)
        required_params: tuple[str] = ("param1", "param_shared")

    @chex.dataclass(frozen=True)
    class MockParams2(BaseConfiguration):
        param2: float = 2.0
        param_shared: float = 3.0
        params_to_optimize: tuple[str] = ("*",)
        required_params: tuple[str] = ("param2", "param_shared")

    @chex.dataclass(frozen=True)
    class MockEnergyFunction1(base.BaseEnergyFunction):
        params: MockParams1

        def compute_energy(self, nucleotide) -> float:
            return nucleotide.center.sum()

    @chex.dataclass(frozen=True)
    class MockEnergyFunction2(base.BaseEnergyFunction):
        params: MockParams2

        def compute_energy(self, nucleotide) -> float:
            return nucleotide.center.sum()

    return [
        MockEnergyFunction1(
            displacement_fn=MagicMock(),
            transform_fn=None,
            topology=MagicMock(),
            params=MockParams1(),
        ),
        MockEnergyFunction2(
            displacement_fn=MagicMock(),
            transform_fn=None,
            topology=MagicMock(),
            params=MockParams2(),
        ),
    ]


def test_composed_energy_function_params_interactions(list_of_energy_functions):
    cef = base.ComposedEnergyFunction(energy_fns=list_of_energy_functions)
    assert cef.params_dict() == {"param1": 1.0, "param2": 2.0, "param_shared": 3.0}
    assert cef.opt_params() == {"param1": 1.0, "param2": 2.0, "param_shared": 3.0}
    assert cef.with_noopt("param1").opt_params() == {"param2": 2.0, "param_shared": 3.0}

    cef_updated = cef.with_params({"param1": 10.0, "param_shared": 100.0})
    assert cef_updated.params_dict() == {"param1": 10.0, "param2": 2.0, "param_shared": 100.0}
    assert all(fn.params.param_shared == 100.0 for fn in cef_updated.energy_fns)

    with pytest.raises(ValueError, match="not used in any"):
        cef.with_params({"non_existent_param": 5.0})

    with pytest.raises(ValueError, match="not used in any"):
        cef.with_params(non_existent_param=5.0)


def test_qualified_composed_energy_function_params_interactions(list_of_energy_functions):
    qualified_cef = base.QualifiedComposedEnergyFunction(energy_fns=list_of_energy_functions)
    assert qualified_cef.params_dict() == {
        "MockEnergyFunction1.param1": 1.0,
        "MockEnergyFunction2.param2": 2.0,
        "MockEnergyFunction1.param_shared": 3.0,
        "MockEnergyFunction2.param_shared": 3.0,
    }
    assert qualified_cef.opt_params() == {
        "MockEnergyFunction1.param1": 1.0,
        "MockEnergyFunction2.param2": 2.0,
        "MockEnergyFunction1.param_shared": 3.0,
        "MockEnergyFunction2.param_shared": 3.0,
    }
    assert qualified_cef.with_noopt("MockEnergyFunction1.param1").opt_params() == {
        "MockEnergyFunction2.param2": 2.0,
        "MockEnergyFunction1.param_shared": 3.0,
        "MockEnergyFunction2.param_shared": 3.0,
    }

    cef_updated = qualified_cef.with_params(
        {
            "MockEnergyFunction1.param1": 10.0,
            "MockEnergyFunction2.param_shared": 100.0,
        }
    )
    assert cef_updated.params_dict() == {
        "MockEnergyFunction1.param1": 10.0,
        "MockEnergyFunction2.param2": 2.0,
        "MockEnergyFunction1.param_shared": 3.0,
        "MockEnergyFunction2.param_shared": 100.0,
    }


def test_composed_energy_function_prop_replacement(list_of_energy_functions):
    cef = base.ComposedEnergyFunction(energy_fns=list_of_energy_functions)
    with_props = cef.with_props(unbonded_neighbors="TEST")
    assert all(fn.unbonded_neighbors == "TEST" for fn in with_props.energy_fns)

    with pytest.raises(ValueError, match="got unexpected kwargs"):
        cef.with_props(non_existent_prop="TEST")


def test_composed_energy_function_without_terms(list_of_energy_functions):
    cef = base.ComposedEnergyFunction(energy_fns=list_of_energy_functions)
    first_by_class = list_of_energy_functions[0].__class__
    without_fn1 = cef.without_terms(first_by_class)
    assert len(without_fn1.energy_fns) == 1
    assert without_fn1.energy_fns == [list_of_energy_functions[1]]

    first_by_name = "MockEnergyFunction1"
    without_fn1 = cef.without_terms(first_by_name)
    assert len(without_fn1.energy_fns) == 1
    assert without_fn1.energy_fns == [list_of_energy_functions[1]]

    cef_w = base.ComposedEnergyFunction(energy_fns=list_of_energy_functions, weights=np.array([2.0, 3.0]))
    without_fn1_w = cef_w.without_terms(first_by_name)
    assert len(without_fn1_w.energy_fns) == 1
    assert without_fn1_w.weights[0] == 3.0
    assert without_fn1_w.weights.shape == (1,)


def test_energy_function_info_initialized_from_topology():
    @chex.dataclass(frozen=True)
    class Top:
        bonded_neighbors: jnp.ndarray
        unbonded_neighbors: jnp.ndarray
        seq: str

    top = Top(bonded_neighbors=jnp.array([1, 2, 3]), unbonded_neighbors=jnp.array([[4, 5, 6]]).T, seq="SEQ")
    ef = MockEnergyFunction(params=None, displacement_fn=MagicMock(), transform_fn=None, topology=top)
    assert jnp.all(ef.bonded_neighbors == top.bonded_neighbors)
    assert jnp.all(ef.unbonded_neighbors == top.unbonded_neighbors.T)
    assert ef.seq == top.seq

    with pytest.raises(ValueError, match="topology"):
        MockEnergyFunction(params=None, displacement_fn=MagicMock(), transform_fn=None)


if __name__ == "__main__":
    pass
