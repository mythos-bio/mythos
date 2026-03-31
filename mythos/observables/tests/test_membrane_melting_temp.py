"""Tests for the membrane melting temperature observable."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import jax_md
import numpy as np

from mythos.observables.membrane_melting_temp import (
    MembraneMeltingTemp,
    apl_residual,
    calculate_apl,
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
        temperature=temp_labels,
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
            return compute_membrane_tm(apls, TEMPS)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(TRUE_APLS)
        assert grads.shape == TRUE_APLS.shape
        # Gradients should be finite
        assert jnp.all(jnp.isfinite(grads))


class TestMembraneMeltingTempBatched:
    """Tests for the batched-by-temperature __call__ path.

    These mock AreaPerLipid to avoid requiring a real GROMACS topology while
    verifying that the per-temperature batching produces correct results.
    """

    def _build_trajectory_and_apls(self, n_per_temp, temps):
        """Build a trajectory and matching per-frame APL values."""
        temps_arr = jnp.asarray(temps)
        true_apls = calculate_apl(
            temps_arr, TRUE_APL0, TRUE_C_P_G, TRUE_DAPL, TRUE_K, TRUE_TM,
        )
        # Each temperature gets n_per_temp identical frames
        trajectories = [_make_trajectory(n_per_temp, (float(t),)) for t in temps_arr]
        combined = SimulatorTrajectory.concat(trajectories)
        per_frame_apls = jnp.repeat(true_apls, n_per_temp)
        return combined, per_frame_apls

    def test_batched_call_uniform_weights(self):
        """Batched __call__ recovers Tm with uniform weights."""
        temps = TEMPS
        n_per_temp = 5
        combined, per_frame_apls = self._build_trajectory_and_apls(n_per_temp, temps)

        # Mock AreaPerLipid.__call__ to return the correct slice of APLs.
        # The observable calls apl_fn(trajectory.slice(indices)) once per
        # temperature; we return the matching APLs by tracking a call counter.
        call_count = [0]
        n_temps = len(temps)

        def fake_apl_call(self_apl, traj):
            idx = call_count[0]
            start = idx * n_per_temp
            end = start + n_per_temp
            call_count[0] += 1
            return per_frame_apls[start:end]

        obs = MembraneMeltingTemp(
            topology=None,  # not used by mock
            lipid_sel="",
            temperatures=temps,
        )

        with patch(
            "mythos.observables.membrane_melting_temp.AreaPerLipid.__call__",
            fake_apl_call,
        ):
            tm = obs(combined)

        np.testing.assert_allclose(tm, TRUE_TM, atol=0.5)
        assert call_count[0] == n_temps

    def test_batched_call_nonuniform_weights(self):
        """Batched __call__ correctly applies per-frame weights."""
        temps = jnp.array([300.0, 310.0, 320.0, 330.0, 340.0])
        temps_arr = temps
        true_apls = calculate_apl(
            temps_arr, TRUE_APL0, TRUE_C_P_G, TRUE_DAPL, TRUE_K, TRUE_TM,
        )
        n_per_temp = 2

        # Two frames per temp: first = true APL, second = true APL + 2
        all_apls = []
        for apl_val in true_apls:
            all_apls.extend([float(apl_val), float(apl_val) + 2.0])
        all_apls = jnp.array(all_apls)

        trajectories = [_make_trajectory(n_per_temp, (float(t),)) for t in temps]
        combined = SimulatorTrajectory.concat(trajectories)

        # Weight toward first frame → should recover original APLs
        weights = jnp.tile(jnp.array([1.0, 1e-12]), len(temps))

        call_count = [0]

        def fake_apl_call(self_apl, traj):
            idx = call_count[0]
            start = idx * n_per_temp
            end = start + n_per_temp
            call_count[0] += 1
            return all_apls[start:end]

        obs = MembraneMeltingTemp(
            topology=None,
            lipid_sel="",
            temperatures=temps,
        )

        with patch(
            "mythos.observables.membrane_melting_temp.AreaPerLipid.__call__",
            fake_apl_call,
        ):
            tm = obs(combined, weights=weights)

        # Weighting toward the "true" frames should recover~TRUE_TM
        np.testing.assert_allclose(tm, TRUE_TM, atol=0.5)
