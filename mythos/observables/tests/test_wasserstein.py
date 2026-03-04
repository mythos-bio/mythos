"""Tests for the Wasserstein distance module."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import wasserstein_distance as scipy_wasserstein

from mythos.observables.base import BaseObservable
from mythos.observables.wasserstein import (
    WassersteinDistance,
    WassersteinDistanceMapped,
    _compute_wasserstein_distance,
    wasserstein_1d,
)
from mythos.simulators.io import SimulatorTrajectory

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - common jax practice


# ---------------------------------------------------------------------------
# Mock observable helpers
# ---------------------------------------------------------------------------

@chex.dataclass(frozen=True, kw_only=True)
class _MockArrayObservable(BaseObservable):
    """Observable that returns a fixed array, ignoring the trajectory."""

    values: jnp.ndarray

    def __call__(self, _trajectory: SimulatorTrajectory) -> jnp.ndarray:
        return self.values


@chex.dataclass(frozen=True, kw_only=True)
class _MockMappedObservable(BaseObservable):
    """Observable that returns a fixed dict[str, ndarray], ignoring the trajectory."""

    value_map: dict

    def __call__(self, _trajectory: SimulatorTrajectory) -> dict[str, jnp.ndarray]:
        return self.value_map


def _dummy_trajectory() -> SimulatorTrajectory:
    """Create a minimal SimulatorTrajectory (unused by mock observables)."""
    centers = jnp.zeros((1, 1, 3))
    orientations = jnp.array([[[1.0, 0.0, 0.0, 0.0]]])
    return SimulatorTrajectory(
        center=centers,
        orientation=orientations,
        box_size=jnp.array([10.0, 10.0, 10.0]),
    )


# ===================================================================
# Tests for wasserstein_1d
# ===================================================================

class TestWasserstein1D:
    """Unit tests for the standalone wasserstein_1d function."""

    def test_identical_distributions_give_zero(self):
        """Distance between a distribution and itself should be zero."""
        u = jnp.array([1.0, 2.0, 3.0])
        assert wasserstein_1d(u, u) == pytest.approx(0.0, abs=1e-12)

    def test_matches_scipy_uniform_weights(self):
        """Result should agree with scipy for uniform-weighted samples."""
        rng = np.random.default_rng(42)
        u = rng.normal(0.0, 1.0, size=50)
        v = rng.normal(1.0, 1.0, size=60)
        expected = scipy_wasserstein(u, v)
        result = float(wasserstein_1d(jnp.array(u), jnp.array(v)))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_matches_scipy_weighted(self):
        """Result should agree with scipy when explicit weights are given."""
        rng = np.random.default_rng(7)
        u = rng.uniform(0, 5, size=30)
        v = rng.uniform(2, 7, size=40)
        u_w = rng.dirichlet(np.ones(30))
        v_w = rng.dirichlet(np.ones(40))
        expected = scipy_wasserstein(u, v, u_weights=u_w, v_weights=v_w)
        result = float(
            wasserstein_1d(
                jnp.array(u), jnp.array(v),
                u_weights=jnp.array(u_w), v_weights=jnp.array(v_w),
            )
        )
        assert result == pytest.approx(expected, rel=1e-6)

    def test_simple_known_value(self):
        """W1 between point masses at 0 and 1 is 1."""
        u = jnp.array([0.0])
        v = jnp.array([1.0])
        assert float(wasserstein_1d(u, v)) == pytest.approx(1.0, abs=1e-12)

    def test_symmetric(self):
        """W1(u, v) == W1(v, u)."""
        rng = np.random.default_rng(99)
        u = jnp.array(rng.normal(size=40))
        v = jnp.array(rng.normal(size=40))
        d1 = float(wasserstein_1d(u, v))
        d2 = float(wasserstein_1d(v, u))
        assert d1 == pytest.approx(d2, rel=1e-10)


# ===================================================================
# Tests for _compute_wasserstein_distance
# ===================================================================

class TestComputeWassersteinDistance:
    """Tests for the internal _compute_wasserstein_distance helper."""

    def test_1d_input(self):
        """Works with already-flat input."""
        u = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([4.0, 5.0, 6.0])
        result = float(_compute_wasserstein_distance(u, v))
        expected = float(wasserstein_1d(u, v))
        assert result == pytest.approx(expected, abs=1e-12)

    def test_2d_input_flattened(self):
        """Multi-dimensional observable output is flattened before comparison."""
        u_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        v = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = float(_compute_wasserstein_distance(u_2d, v))
        expected = float(wasserstein_1d(u_2d.flatten(), v))
        assert result == pytest.approx(expected, abs=1e-12)

    def test_with_weights(self):
        """Weights are broadcast when observable output is multi-dimensional."""
        # 2 states, each producing a single value  →  obs shape (2,)
        u = jnp.array([0.0, 10.0])
        v = jnp.array([5.0])
        weights = jnp.array([0.3, 0.7])
        result = float(_compute_wasserstein_distance(u, v, weights=weights))
        expected = float(wasserstein_1d(u, v, u_weights=weights))
        assert result == pytest.approx(expected, abs=1e-12)


# ===================================================================
# Tests for WassersteinDistance
# ===================================================================

class TestWassersteinDistance:
    """Tests for the WassersteinDistance callable dataclass."""

    def test_basic_distance(self):
        u_vals = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([4.0, 5.0, 6.0])
        obs = _MockArrayObservable(
            rigid_body_transform_fn=lambda x: x,
            values=u_vals,
        )
        wd = WassersteinDistance(
            observable=obs,
            v_distribution=v,
        )
        result = float(wd(_dummy_trajectory()))
        expected = float(wasserstein_1d(u_vals, v))
        assert result == pytest.approx(expected, abs=1e-12)

    def test_with_v_weights(self):
        u_vals = jnp.array([0.0, 1.0])
        v = jnp.array([0.5, 1.5, 2.5])
        v_w = jnp.array([0.5, 0.3, 0.2])
        obs = _MockArrayObservable(
            rigid_body_transform_fn=lambda x: x,
            values=u_vals,
        )
        wd = WassersteinDistance(
            observable=obs,
            v_distribution=v,
            v_weights=v_w,
        )
        result = float(wd(_dummy_trajectory()))
        expected = float(wasserstein_1d(u_vals, v, v_weights=v_w))
        assert result == pytest.approx(expected, abs=1e-12)

    def test_with_u_weights(self):
        u_vals = jnp.array([0.0, 1.0, 2.0])
        v = jnp.array([0.0, 1.0, 2.0])
        u_w = jnp.array([0.1, 0.2, 0.7])
        obs = _MockArrayObservable(
            rigid_body_transform_fn=lambda x: x,
            values=u_vals,
        )
        wd = WassersteinDistance(
            observable=obs,
            v_distribution=v,
        )
        result = float(wd(_dummy_trajectory(), weights=u_w))
        expected = float(wasserstein_1d(u_vals, v, u_weights=u_w))
        assert result == pytest.approx(expected, abs=1e-12)

    def test_identical_distributions(self):
        vals = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        obs = _MockArrayObservable(
            rigid_body_transform_fn=lambda x: x,
            values=vals,
        )
        wd = WassersteinDistance(
            observable=obs,
            v_distribution=vals,
        )
        assert float(wd(_dummy_trajectory())) == pytest.approx(0.0, abs=1e-12)


# ===================================================================
# Tests for WassersteinDistanceMapped
# ===================================================================

class TestWassersteinDistanceMapped:
    """Tests for the WassersteinDistanceMapped callable dataclass."""

    def test_single_key(self):
        """With a single key the result dict has one entry matching plain WD."""
        u_map = {"bond_AB": jnp.array([1.0, 2.0, 3.0])}
        v_map = {"bond_AB": jnp.array([2.0, 3.0, 4.0])}
        obs = _MockMappedObservable(
            rigid_body_transform_fn=lambda x: x,
            value_map=u_map,
        )
        wdm = WassersteinDistanceMapped(
            observable=obs,
            v_distribution_map=v_map,
        )
        result = wdm(_dummy_trajectory())
        assert set(result.keys()) == {"bond_AB"}
        expected = float(wasserstein_1d(u_map["bond_AB"], v_map["bond_AB"]))
        assert float(result["bond_AB"]) == pytest.approx(expected, abs=1e-12)

    def test_multiple_keys(self):
        """Each key is computed independently."""
        u_map = {
            "bond_AB": jnp.array([0.0, 1.0]),
            "bond_CD": jnp.array([10.0, 20.0, 30.0]),
        }
        v_map = {
            "bond_AB": jnp.array([0.5, 1.5]),
            "bond_CD": jnp.array([15.0, 25.0]),
        }
        obs = _MockMappedObservable(
            rigid_body_transform_fn=lambda x: x,
            value_map=u_map,
        )
        wdm = WassersteinDistanceMapped(
            observable=obs,
            v_distribution_map=v_map,
        )
        result = wdm(_dummy_trajectory())
        assert set(result.keys()) == {"bond_AB", "bond_CD"}
        for key in v_map:
            expected = float(wasserstein_1d(u_map[key], v_map[key]))
            assert float(result[key]) == pytest.approx(expected, abs=1e-12)

    def test_with_v_weights_map(self):
        """Per-key V weights are forwarded correctly."""
        u_map = {
            "angle_X": jnp.array([1.0, 2.0, 3.0]),
            "angle_Y": jnp.array([5.0, 6.0]),
        }
        v_map = {
            "angle_X": jnp.array([1.5, 2.5]),
            "angle_Y": jnp.array([5.5, 6.5, 7.5]),
        }
        v_w_map = {
            "angle_X": jnp.array([0.6, 0.4]),
            # angle_Y intentionally omitted → defaults to uniform
        }
        obs = _MockMappedObservable(
            rigid_body_transform_fn=lambda x: x,
            value_map=u_map,
        )
        wdm = WassersteinDistanceMapped(
            observable=obs,
            v_distribution_map=v_map,
            v_weights_map=v_w_map,
        )
        result = wdm(_dummy_trajectory())

        # angle_X: explicit v-weights
        exp_x = float(wasserstein_1d(
            u_map["angle_X"], v_map["angle_X"],
            v_weights=v_w_map["angle_X"],
        ))
        assert float(result["angle_X"]) == pytest.approx(exp_x, abs=1e-12)

        # angle_Y: uniform v-weights (not in v_weights_map)
        exp_y = float(wasserstein_1d(u_map["angle_Y"], v_map["angle_Y"]))
        assert float(result["angle_Y"]) == pytest.approx(exp_y, abs=1e-12)

    def test_with_shared_u_weights(self):
        """U weights are applied identically to every key."""
        u_map = {
            "k1": jnp.array([0.0, 1.0]),
            "k2": jnp.array([2.0, 3.0]),
        }
        v_map = {
            "k1": jnp.array([0.5]),
            "k2": jnp.array([2.5]),
        }
        u_w = jnp.array([0.3, 0.7])
        obs = _MockMappedObservable(
            rigid_body_transform_fn=lambda x: x,
            value_map=u_map,
        )
        wdm = WassersteinDistanceMapped(
            observable=obs,
            v_distribution_map=v_map,
        )
        result = wdm(_dummy_trajectory(), weights=u_w)
        for key in v_map:
            expected = float(_compute_wasserstein_distance(
                u_map[key], v_map[key], weights=u_w,
            ))
            assert float(result[key]) == pytest.approx(expected, abs=1e-12)

    def test_identical_distributions_mapped(self):
        """All distances should be zero when U == V per key."""
        shared = {
            "a": jnp.array([1.0, 2.0, 3.0]),
            "b": jnp.array([4.0, 5.0]),
        }
        obs = _MockMappedObservable(
            rigid_body_transform_fn=lambda x: x,
            value_map=shared,
        )
        wdm = WassersteinDistanceMapped(
            observable=obs,
            v_distribution_map=shared,
        )
        result = wdm(_dummy_trajectory())
        for key in shared:
            assert float(result[key]) == pytest.approx(0.0, abs=1e-12)

    def test_keys_match_v_distribution_map(self):
        """Output keys are exactly the keys of v_distribution_map."""
        u_map = {
            "x": jnp.array([1.0]),
            "y": jnp.array([2.0]),
            "z": jnp.array([3.0]),  # extra key not in v_distribution_map
        }
        v_map = {
            "x": jnp.array([1.0]),
            "y": jnp.array([2.0]),
        }
        obs = _MockMappedObservable(
            rigid_body_transform_fn=lambda x: x,
            value_map=u_map,
        )
        wdm = WassersteinDistanceMapped(
            observable=obs,
            v_distribution_map=v_map,
        )
        result = wdm(_dummy_trajectory())
        assert set(result.keys()) == set(v_map.keys())
