"""Tests for the membrane melting temperature observable."""

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

from mythos.observables.membrane_melting_temp import (
    apl_residual,
    build_segment_ids,
    calculate_apl,
    compute_expected_apls,
    compute_membrane_tm,
    fit_apl_sigmoid,
    get_initial_guess,
)
from mythos.simulators.io import SimulatorTrajectory

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - common jax practice

TRUE_APL0 = 47.0
TRUE_C_P_G = 0.01
TRUE_DAPL = 8.0
TRUE_K = 0.3
TRUE_TM = 320.0

TEMPS = jnp.linspace(290.0, 350.0, 13)
TRUE_APLS = calculate_apl(TEMPS, TRUE_APL0, TRUE_C_P_G, TRUE_DAPL, TRUE_K, TRUE_TM)


def _make_trajectory(
    n_frames_per_temp: int,
    temperatures: tuple[float, ...],
) -> SimulatorTrajectory:
    """Create a minimal trajectory with temperature metadata."""
    n_temps = len(temperatures)
    total = n_frames_per_temp * n_temps
    n_atoms = 4  # dummy

    centers = jnp.zeros((total, n_atoms, 3))
    quats = jnp.tile(
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        (total, n_atoms, 1),
    )
    box_size = jnp.full((total, 3), 10.0)

    # Build per-frame temperature labels via repeat
    temp_labels = jnp.repeat(jnp.array(temperatures), n_frames_per_temp)

    return SimulatorTrajectory(
        center=centers,
        orientation=jax_md.rigid_body.Quaternion(vec=quats),
        box_size=box_size,
        metadata={"temp": temp_labels},
    )


class TestCalculateApl:
    """Tests for the calculate_apl sigmoid model function."""

    def test_known_value_at_tm(self):
        """At T=Tm the sigmoid term is dAPL/2."""
        result = calculate_apl(TRUE_TM, TRUE_APL0, TRUE_C_P_G, TRUE_DAPL, TRUE_K, TRUE_TM)
        expected = TRUE_APL0 + TRUE_C_P_G * TRUE_TM + TRUE_DAPL / 2
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_low_temp_near_baseline(self):
        """Far below Tm the sigmoid contribution is near zero."""
        result = calculate_apl(200.0, TRUE_APL0, TRUE_C_P_G, TRUE_DAPL, TRUE_K, TRUE_TM)
        # sigmoid ≈ 0 at T << Tm
        expected = TRUE_APL0 + TRUE_C_P_G * 200.0
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_high_temp_near_plateau(self):
        """Far above Tm the sigmoid contribution is near dAPL."""
        result = calculate_apl(500.0, TRUE_APL0, TRUE_C_P_G, TRUE_DAPL, TRUE_K, TRUE_TM)
        expected = TRUE_APL0 + TRUE_C_P_G * 500.0 + TRUE_DAPL
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_vectorized(self):
        """Should work on arrays of temperatures."""
        result = calculate_apl(TEMPS, TRUE_APL0, TRUE_C_P_G, TRUE_DAPL, TRUE_K, TRUE_TM)
        assert result.shape == TEMPS.shape


class TestAplResidual:
    """Tests for the least-squares residual function."""

    def test_zero_residual_at_true_params(self):
        """Residual should be zero when coefficients match the generating parameters."""
        coeffs = jnp.array([TRUE_APL0, TRUE_C_P_G, TRUE_DAPL, TRUE_K, TRUE_TM])
        residuals = apl_residual(coeffs, (TRUE_APLS, TEMPS))
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


class TestGetInitialGuess:
    """Tests for the heuristic initial guess."""

    def test_reasonable_guess(self):
        """Tm guess should fall within the temperature range."""
        guess = get_initial_guess(TRUE_APLS, TEMPS)
        assert guess.shape == (5,)
        # Tm guess is the median temperature
        assert TEMPS.min() <= guess[4] <= TEMPS.max()

    def test_positive_dapl(self):
        """dAPL should be positive for increasing APL with temperature."""
        guess = get_initial_guess(TRUE_APLS, TEMPS)
        assert guess[2] > 0


class TestFitAplSigmoid:
    """Tests for the nonlinear least-squares sigmoid fit."""

    def test_recovers_true_params(self):
        """Fit should recover the generating parameters from clean data."""
        fitted = fit_apl_sigmoid(TRUE_APLS, TEMPS)
        # Tm is the most important parameter
        np.testing.assert_allclose(fitted[4], TRUE_TM, atol=0.5)
        # dAPL
        np.testing.assert_allclose(fitted[2], TRUE_DAPL, atol=0.5)

    def test_residuals_small(self):
        """Residuals after fitting should be small."""
        fitted = fit_apl_sigmoid(TRUE_APLS, TEMPS)
        residuals = apl_residual(fitted, (TRUE_APLS, TEMPS))
        assert jnp.max(jnp.abs(residuals)) < 0.01


class TestComputeMembraneTm:
    """Tests for the convenience Tm extraction function."""

    def test_returns_correct_tm(self):
        """Should return Tm close to the true value."""
        tm = compute_membrane_tm(TRUE_APLS, TEMPS)
        np.testing.assert_allclose(tm, TRUE_TM, atol=0.5)

    def test_differentiable_wrt_apls(self):
        """jax.grad should work through the solver (implicit diff)."""
        def loss_fn(apls):
            return compute_membrane_tm(apls, TEMPS, implicit_diff=True)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(TRUE_APLS)
        assert grads.shape == TRUE_APLS.shape
        # Gradients should be finite
        assert jnp.all(jnp.isfinite(grads))


class TestBuildSegmentIds:
    """Tests for temperature-to-segment-id mapping."""

    def test_exact_match(self):
        """Exact temperature values should map to the correct index."""
        temps = jnp.array([300.0, 310.0, 320.0])
        labels = jnp.array([310.0, 300.0, 320.0, 310.0])
        ids = build_segment_ids(labels, temps)
        np.testing.assert_array_equal(ids, jnp.array([1, 0, 2, 1]))

    def test_closest_match(self):
        """Labels near a temperature should round to the closest index."""
        temps = jnp.array([300.0, 320.0])
        labels = jnp.array([299.9, 320.1])
        ids = build_segment_ids(labels, temps)
        np.testing.assert_array_equal(ids, jnp.array([0, 1]))

    def test_rejects_unmatched_temperatures(self):
        """Labels far from any temperature should raise ValueError."""
        temps = jnp.array([300.0, 320.0, 340.0])
        labels = jnp.array([300.0, 315.0, 340.0])  # 315 is 5K from nearest
        with pytest.raises(ValueError, match="do not match"):
            build_segment_ids(labels, temps)

    def test_custom_atol(self):
        """Custom atol allows larger deviations when appropriate."""
        temps = jnp.array([300.0, 320.0])
        labels = jnp.array([302.0, 318.0])  # 2K off
        # Default atol=1.0 should reject
        with pytest.raises(ValueError):
            build_segment_ids(labels, temps)
        # Larger atol should accept
        ids = build_segment_ids(labels, temps, atol=3.0)
        np.testing.assert_array_equal(ids, jnp.array([0, 1]))


class TestComputeExpectedApls:
    """Tests for segment-sum weighted expected APL."""

    def test_uniform_weights_gives_mean(self):
        """With uniform weights, expected APL equals arithmetic mean per group."""
        apls = jnp.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        weights = jnp.ones(6)
        segment_ids = jnp.array([0, 0, 0, 1, 1, 1])
        result = compute_expected_apls(apls, weights, segment_ids, n_segments=2)
        np.testing.assert_allclose(result, jnp.array([2.0, 20.0]))

    def test_weighted(self):
        """Non-uniform weights should shift the expected APL."""
        apls = jnp.array([1.0, 3.0])
        weights = jnp.array([0.75, 0.25])
        segment_ids = jnp.array([0, 0])
        result = compute_expected_apls(apls, weights, segment_ids, n_segments=1)
        expected = (0.75 * 1.0 + 0.25 * 3.0) / (0.75 + 0.25)
        np.testing.assert_allclose(result[0], expected)

    def test_differentiable_wrt_weights(self):
        """Gradient with respect to weights should be finite."""
        apls = jnp.array([1.0, 3.0, 5.0, 7.0])
        segment_ids = jnp.array([0, 0, 1, 1])

        def fn(w):
            return jnp.sum(compute_expected_apls(apls, w, segment_ids, n_segments=2))

        grads = jax.grad(fn)(jnp.ones(4))
        assert jnp.all(jnp.isfinite(grads))


class TestMembraneMeltingTempObservable:
    """Tests for the full observable class (module-function path only).

    These tests exercise the segment/fitting pipeline with synthetic data
    rather than calling the AreaPerLipid observable (which requires a real
    GROMACS topology).  They verify the class wiring via
    ``compute_expected_apls`` → ``compute_membrane_tm``.
    """

    def test_compute_pipeline_uniform_weights(self):
        """End-to-end: synthetic per-frame APLs with uniform weights."""
        n_per_temp = 5
        temps = tuple(float(t) for t in TEMPS)
        n_temps = len(temps)
        temps_arr = jnp.array(temps)

        # Simulate per-frame APLs: each frame at a temperature gets the
        # sigmoid value (no noise)
        per_frame_apls = jnp.repeat(TRUE_APLS, n_per_temp)
        temp_labels = jnp.repeat(temps_arr, n_per_temp)
        weights = jnp.ones(n_per_temp * n_temps)

        segment_ids = build_segment_ids(temp_labels, temps_arr)
        expected_apls = compute_expected_apls(per_frame_apls, weights, segment_ids, n_temps)
        tm = compute_membrane_tm(expected_apls, temps_arr)
        np.testing.assert_allclose(tm, TRUE_TM, atol=0.5)

    def test_compute_pipeline_nonuniform_weights(self):
        """Weighted APLs shift the effective curve and thus Tm."""
        temps = (300.0, 310.0, 320.0, 330.0, 340.0)
        temps_arr = jnp.array(temps)
        true_apls = calculate_apl(temps_arr, TRUE_APL0, TRUE_C_P_G, TRUE_DAPL, TRUE_K, TRUE_TM)

        # Two frames per temperature; second frame has APL + 2.0
        n_per_temp = 2
        per_frame_apls = []
        for apl_val in true_apls:
            per_frame_apls.extend([apl_val, apl_val + 2.0])
        per_frame_apls = jnp.array(per_frame_apls)
        temp_labels = jnp.repeat(temps_arr, n_per_temp)

        # Uniform weights → expected APL = mean = apl_val + 1.0
        weights_uniform = jnp.ones(len(per_frame_apls))
        segment_ids = build_segment_ids(temp_labels, temps_arr)
        expected_uniform = compute_expected_apls(
            per_frame_apls, weights_uniform, segment_ids, len(temps)
        )
        np.testing.assert_allclose(expected_uniform, true_apls + 1.0, atol=1e-10)

        # Fully weight toward the first frame of each pair → should recover original
        weights_first = jnp.tile(jnp.array([1.0, 0.0]), len(temps))
        # Need to handle zero weights: add tiny epsilon
        weights_first = weights_first + 1e-12
        expected_first = compute_expected_apls(
            per_frame_apls, weights_first, segment_ids, len(temps)
        )
        np.testing.assert_allclose(expected_first, true_apls, atol=1e-4)

    def test_metadata_preserved_through_concat(self):
        """Temperature metadata survives SimulatorTrajectory.concat."""
        traj1 = _make_trajectory(3, (300.0,))
        traj2 = _make_trajectory(3, (320.0,))
        combined = SimulatorTrajectory.concat([traj1, traj2])

        expected_labels = jnp.array([300.0, 300.0, 300.0, 320.0, 320.0, 320.0])
        np.testing.assert_allclose(combined.metadata["temp"], expected_labels)

    def test_segment_ids_from_concat(self):
        """Segment ids correctly map concatenated trajectory frames."""
        traj1 = _make_trajectory(2, (300.0,))
        traj2 = _make_trajectory(2, (320.0,))
        combined = SimulatorTrajectory.concat([traj1, traj2])

        temps_arr = jnp.array([300.0, 320.0])
        ids = build_segment_ids(combined.metadata["temp"].squeeze(), temps_arr)
        np.testing.assert_array_equal(ids, jnp.array([0, 0, 1, 1]))
