import chex
import jax.numpy as jnp
import jax_md
import jax_md.space
import numpy as np
import pytest

import mythos.observables.base as b
import mythos.observables.stretch_torsion as st
import mythos.simulators.io as jd_sio

# Use jax_md.space.free for displacement function
displacement_fn, _ = jax_md.space.free()


class TestSingleAngleXY:

    def test_parallel_base_pairs_zero_angle(self) -> None:
        # Two base pairs pointing in the same direction in the XY plane
        base_sites = jnp.array([
            [0.0, 0.0, 0.0],  # a1
            [1.0, 0.0, 0.0],  # b1
            [0.0, 0.0, 1.0],  # a2
            [1.0, 0.0, 1.0],  # b2
        ])

        result = st.single_angle_xy(jnp.array([[0, 1], [2, 3]]), base_sites, displacement_fn)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_perpendicular_base_pairs_right_angle(self) -> None:
        # Two base pairs at 90 degrees in the XY plane
        base_sites = jnp.array([
            [0.0, 0.0, 0.0],  # a1
            [1.0, 0.0, 0.0],  # b1 - vector along +x
            [0.0, 0.0, 1.0],  # a2
            [0.0, 1.0, 1.0],  # b2 - vector along +y
        ])

        result = st.single_angle_xy(jnp.array([[0, 1], [2, 3]]), base_sites, displacement_fn)
        np.testing.assert_allclose(result, jnp.pi / 2, atol=1e-6)

    def test_antiparallel_base_pairs_pi_angle(self) -> None:
        # Two base pairs pointing in opposite directions in the XY plane
        base_sites = jnp.array([
            [0.0, 0.0, 0.0],  # a1
            [1.0, 0.0, 0.0],  # b1 - vector along +x
            [0.0, 0.0, 1.0],  # a2
            [-1.0, 0.0, 1.0],  # b2 - vector along -x
        ])

        result = st.single_angle_xy(jnp.array([[0, 1], [2, 3]]), base_sites, displacement_fn)
        np.testing.assert_allclose(result, jnp.pi, atol=1e-6)


class TestTwistXY:

    def test_post_init_raises_without_transform_fn(self) -> None:
        with pytest.raises(ValueError, match=b.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED):
            st.TwistXY(
                rigid_body_transform_fn=None,
                quartets=jnp.array([[[0, 1], [2, 3]]]),
                displacement_fn=displacement_fn,
            )

    def test_twist_call(self) -> None:

        @chex.dataclass
        class MockNucleotide:
            base_sites: jnp.ndarray

            @staticmethod
            def from_rigid_body(rigid_body: jax_md.rigid_body.RigidBody) -> "MockNucleotide":
                return MockNucleotide(base_sites=rigid_body.center)

        def mock_rbt(x: jax_md.rigid_body.RigidBody) -> MockNucleotide:
            return MockNucleotide.from_rigid_body(x)

        t = 3
        # Base sites: pairs parallel in XY plane -> zero angle contributions
        centers = jnp.array([
            [
                [0.0, 0.0, 0.0],  # a1
                [1.0, 0.0, 0.0],  # b1
                [0.0, 0.0, 1.0],  # a2
                [1.0, 0.0, 1.0],  # b2
            ]
        ] * t)
        orientations = jnp.ones([t, 4, 4])
        trajectory = jd_sio.SimulatorTrajectory(
            center=centers,
            orientation=orientations,
        )

        twist = st.TwistXY(
            rigid_body_transform_fn=mock_rbt,
            quartets=jnp.array([[[0, 1], [2, 3]]]),
            displacement_fn=displacement_fn,
        )

        result = twist(trajectory)
        # All parallel pairs should give 0 total twist
        np.testing.assert_allclose(result, jnp.zeros(t), atol=1e-6)


class TestSingleExtensionZ:

    def test_extension_along_z(self) -> None:
        center = jnp.array([
            [0.0, 0.0, 0.0],  # a1
            [2.0, 0.0, 0.0],  # b1
            [0.0, 0.0, 5.0],  # a2
            [2.0, 0.0, 5.0],  # b2
        ])
        bp1 = jnp.array([0, 1])
        bp2 = jnp.array([2, 3])

        result = st.single_extension_z(center, bp1, bp2, displacement_fn)
        np.testing.assert_allclose(result, 5.0, atol=1e-6)

    def test_extension_negative_z_results_abs(self) -> None:
        center = jnp.array([
            [0.0, 0.0, 10.0],  # a1
            [2.0, 0.0, 10.0],  # b1
            [0.0, 0.0, 3.0],   # a2
            [2.0, 0.0, 3.0],   # b2
        ])
        bp1 = jnp.array([0, 1])
        bp2 = jnp.array([2, 3])

        result = st.single_extension_z(center, bp1, bp2, displacement_fn)
        np.testing.assert_allclose(result, 7.0, atol=1e-6)

    def test_extension_zero_z(self) -> None:
        center = jnp.array([
            [0.0, 0.0, 5.0],  # a1
            [2.0, 0.0, 5.0],  # b1
            [4.0, 0.0, 5.0],  # a2
            [6.0, 0.0, 5.0],  # b2
        ])
        bp1 = jnp.array([0, 1])
        bp2 = jnp.array([2, 3])

        result = st.single_extension_z(center, bp1, bp2, displacement_fn)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)


class TestExtensionZ:

    def test_post_init_raises_without_transform_fn(self) -> None:
        with pytest.raises(ValueError, match=b.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED):
            st.ExtensionZ(
                rigid_body_transform_fn=None,
                bp1=jnp.array([0, 1]),
                bp2=jnp.array([2, 3]),
                displacement_fn=displacement_fn,
            )

    def test_extension_call(self) -> None:

        @chex.dataclass
        class MockNucleotide:
            center: jnp.ndarray

            @staticmethod
            def from_rigid_body(rigid_body: jax_md.rigid_body.RigidBody) -> "MockNucleotide":
                return MockNucleotide(center=rigid_body.center)

        def mock_rbt(x: jax_md.rigid_body.RigidBody) -> MockNucleotide:
            return MockNucleotide.from_rigid_body(x)

        t = 3
        centers = jnp.array([
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
                [2.0, 0.0, 10.0],
            ]
        ] * t)
        orientations = jnp.ones([t, 4, 4])
        trajectory = jd_sio.SimulatorTrajectory(
            center=centers,
            orientation=orientations,
        )

        extension = st.ExtensionZ(
            rigid_body_transform_fn=mock_rbt,
            bp1=jnp.array([0, 1]),
            bp2=jnp.array([2, 3]),
            displacement_fn=displacement_fn,
        )

        result = extension(trajectory)
        np.testing.assert_allclose(result, jnp.full(t, 10.0), atol=1e-6)


class TestStretch:

    def test_stretch_linear_response(self) -> None:
        # Linear relationship: extension = l0 + a1 * force
        # where a1 = 2.0, l0 = 10.0, so s_eff = l0 / a1 = 5.0
        forces = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        extensions = jnp.array([10.0, 12.0, 14.0, 16.0, 18.0])

        a1, l0, s_eff = st.stretch(forces, extensions)

        np.testing.assert_allclose(a1, 2.0, atol=1e-5)
        np.testing.assert_allclose(l0, 10.0, atol=1e-5)
        np.testing.assert_allclose(s_eff, 5.0, atol=1e-5)

    def test_stretch_zero_slope(self) -> None:
        forces = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        extensions = jnp.array([10.0, 10.0, 10.0, 10.0, 10.0])

        a1, l0, s_eff = st.stretch(forces, extensions)

        np.testing.assert_allclose(a1, 0.0, atol=1e-5)
        np.testing.assert_allclose(l0, 10.0, atol=1e-5)
        # s_eff = l0 / a1 will be very large in magnitude when a1 ≈ 0
        # Due to numerical precision, a1 is not exactly 0, so s_eff is large but finite
        assert jnp.abs(s_eff) > 1e5


class TestTorsion:

    def test_torsion_linear_response(self) -> None:
        # extension = ext0 + a3 * torque, where a3 = 0.5
        # twist = tw0 + a4 * torque, where a4 = 1.5
        torques = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        extensions = jnp.array([10.0, 10.5, 11.0, 11.5, 12.0])
        twists = jnp.array([1.0, 2.5, 4.0, 5.5, 7.0])

        a3, a4 = st.torsion(torques, extensions, twists)

        np.testing.assert_allclose(a3, 0.5, atol=1e-5)
        np.testing.assert_allclose(a4, 1.5, atol=1e-5)

    def test_torsion_zero_slope(self) -> None:
        torques = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        extensions = jnp.array([10.0, 10.0, 10.0, 10.0, 10.0])
        twists = jnp.array([5.0, 5.0, 5.0, 5.0, 5.0])

        a3, a4 = st.torsion(torques, extensions, twists)

        np.testing.assert_allclose(a3, 0.0, atol=1e-5)
        np.testing.assert_allclose(a4, 0.0, atol=1e-5)


class TestStretchTorsion:

    def test_stretch_torsion_combined(self) -> None:
        # Set up known linear relationships:
        # force-extension: ext = 10.0 + 2.0 * F -> l0=10, a1=2, s_eff=5
        # torque-extension: ext = 10.0 + 0.5 * T -> a3=0.5
        # torque-twist: twist = 1.0 + 1.5 * T -> a4=1.5

        forces = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        force_extensions = jnp.array([10.0, 12.0, 14.0, 16.0, 18.0])

        torques = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        torque_extensions = jnp.array([10.0, 10.5, 11.0, 11.5, 12.0])
        torque_twists = jnp.array([1.0, 2.5, 4.0, 5.5, 7.0])

        s_eff, c, g = st.stretch_torsion(
            forces, force_extensions, torques, torque_extensions, torque_twists
        )

        # a1=2, l0=10, s_eff=5
        # a3=0.5, a4=1.5
        # c = a1*l0 / (a4*a1 - a3^2) = 2*10 / (1.5*2 - 0.25) = 20 / 2.75 ≈ 7.272727
        # g = -(a3*l0) / (a4*a1 - a3^2) = -(0.5*10) / 2.75 = -5 / 2.75 ≈ -1.818182
        expected_c = 20.0 / 2.75
        expected_g = -5.0 / 2.75

        np.testing.assert_allclose(s_eff, 5.0, atol=1e-5)
        np.testing.assert_allclose(c, expected_c, atol=1e-5)
        np.testing.assert_allclose(g, expected_g, atol=1e-5)

    def test_stretch_torsion_no_coupling(self) -> None:
        forces = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        force_extensions = jnp.array([10.0, 12.0, 14.0, 16.0, 18.0])

        torques = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # Extension doesn't change with torque -> a3 = 0
        torque_extensions = jnp.array([10.0, 10.0, 10.0, 10.0, 10.0])
        torque_twists = jnp.array([1.0, 2.5, 4.0, 5.5, 7.0])

        s_eff, c, g = st.stretch_torsion(
            forces, force_extensions, torques, torque_extensions, torque_twists
        )

        # a1=2, l0=10, s_eff=5
        # a3=0, a4=1.5
        # c = a1*l0 / (a4*a1 - a3^2) = 20 / 3 ≈ 6.666...
        # g = 0 (no coupling)
        expected_c = 20.0 / 3.0

        np.testing.assert_allclose(s_eff, 5.0, atol=1e-5)
        np.testing.assert_allclose(c, expected_c, atol=1e-5)
        np.testing.assert_allclose(g, 0.0, atol=1e-5)
