# keep it first
# due to astropy units/constants initialization
from unittests import set_astropy_units

import numpy as np
from numpy.testing import assert_allclose

from elisa.binary_system.orbit import orbit as bin_orbit
from elisa.single_system.orbit import orbit as single_orbit
from unittests.utils import ElisaTestCase

set_astropy_units()


class TestOrbitHelpers(ElisaTestCase):
    """Tests for binary-orbit helper functions in elisa.binary_system.orbit.orbit."""

    def test_velocity_vector_angle_zero_eccentricity(self) -> None:
        """Basic sanity: e=0, true_anomaly=0 -> sin=1, cos=0."""
        sin, cos = bin_orbit.velocity_vector_angle(0.0, 0.0)
        assert_allclose(sin, 1.0)
        assert_allclose(cos, 0.0)

    def test_distance_to_center_of_mass_simple(self) -> None:
        """Equal masses, distance splits equally."""
        d1, d2 = bin_orbit.distance_to_center_of_mass(1.0, 1.0, 2.0)
        assert_allclose(d1, 1.0)
        assert_allclose(d2, 1.0)

    def test_orbital_semi_major_axes_and_component_distance(self) -> None:
        """Sanity checks for orbital semi-major axis and component distance helpers."""
        r = 1.0
        e = 0.0
        nu = 0.5

        sma = bin_orbit.orbital_semi_major_axes(r, e, nu)
        # with e == 0 the function should return r
        assert_allclose(sma, 1.0)

        comp_dist = bin_orbit.component_distance_from_mean_anomaly(e, nu)
        # with e == 0 should be 1.0
        assert_allclose(comp_dist, 1.0)

    def test_angular_velocity(self) -> None:
        """Sanity test for angular_velocity (static expected value check)."""
        expected = 7.349e-05
        obtained = round(bin_orbit.angular_velocity(1.25, 0.3, 0.869), 8)
        self.assertEqual(expected, obtained)

    def test_create_orb_vel_vectors_basic(self) -> None:
        """Verify returned vectors have expected shapes and symmetry using a small mock system."""

        class MockPosition:
            def __init__(self, true_anomaly: float) -> None:
                self.true_anomaly = true_anomaly

        class MockStar:
            def __init__(self, mass: float) -> None:
                self.mass = mass

        class MockSystem:
            def __init__(self) -> None:
                self.semi_major_axis = 1.0
                self.mass_ratio = 1.0
                self.primary = MockStar(1.0)
                self.secondary = MockStar(1.0)
                self.position = MockPosition(0.0)
                self.eccentricity = 0.0

        sys = MockSystem()
        vectors = bin_orbit.create_orb_vel_vectors(sys, components_distance=1.0)

        self.assertSetEqual(set(vectors.keys()), {"primary", "secondary"})

        primary = vectors["primary"]
        secondary = vectors["secondary"]

        self.assertEqual(primary.shape, (3,))
        self.assertEqual(secondary.shape, (3,))

        # center-of-mass frame: primary + secondary should be approximately zero
        assert_allclose(primary + secondary, np.zeros(3))

        # secondary should equal -primary / mass_ratio (mass_ratio==1)
        assert_allclose(secondary, -primary)


class TestSingleSystemOrbit(ElisaTestCase):
    """Tests for single-system orbit utilities and Orbit.rotational_motion methods."""

    def setUp(self) -> None:
        """Create a simple Orbit instance for testing rotational methods."""
        # supply minimal valid kwargs for single-system Orbit
        self.orbit = single_orbit.Orbit(rotation_period=1.0, inclination=1.0)

    def test_rotational_motion_scalar_and_array(self) -> None:
        """Verify scalar and array inputs produce expected shapes and values."""
        # scalar
        out = self.orbit.rotational_motion(0.0)
        # expect shape (1,3)
        self.assertEqual(out.shape, (1, 3))
        assert_allclose(out[0, 0], single_orbit.true_phase_to_azimuth(0.0))
        assert_allclose(out[0, 2], 0.0)

        # array
        phases = np.array([0.0, 0.25, 0.5])
        out2 = self.orbit.rotational_motion(phases)
        self.assertEqual(out2.shape, (3, 3))
        assert_allclose(out2[:, 0], single_orbit.true_phase_to_azimuth(phases))
        assert_allclose(out2[:, 2], phases)

    def test_rotational_motion_from_azimuths(self) -> None:
        """Verify azimuth -> phase mapping is consistent with azimuth_to_true_phase."""
        az = np.array([0.0, 1.0, 2.0])
        out = self.orbit.rotational_motion_from_azimuths(az)
        self.assertEqual(out.shape, (3, 3))
        assert_allclose(out[:, 0], az)
        assert_allclose(out[:, 2], single_orbit.azimuth_to_true_phase(az))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-q"])
