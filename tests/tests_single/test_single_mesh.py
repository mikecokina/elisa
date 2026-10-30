# keep it first
# due to stupid astropy units/constants implementation
from tests import set_astropy_units

import numpy as np
import pytest

from numpy.testing import assert_array_equal

from tests import utils as testutils
from tests.utils import ElisaTestCase, prepare_single_system

from elisa.base import error
from elisa.single_system import model
from elisa.utils import is_empty, find_nearest_dist_3d
from elisa import umpy as up, units as u

set_astropy_units()


class BuildMeshSpotsFreeTestCase(ElisaTestCase):
    def test_mesh_for_duplicate_points(self):
        for params in testutils.SINGLE_SYSTEM_PARAMS.values():
            s = prepare_single_system(params)
            system_container = testutils.prepare_single_system_container(s)

            system_container.build_mesh()

            star_instance = getattr(system_container, 'star')
            distance = round(find_nearest_dist_3d(list(star_instance.points)), 10)

            if distance < 1e-10:
                bad_points = []
                for i, point in enumerate(star_instance.points):
                    for j, xx in enumerate(star_instance.points[i + 1:]):
                        dist = np.linalg.norm(point - xx)
                        if dist <= 1e-10:
                            print(f'Match: {point}, {i}, {j}')
                            bad_points.append(point)

            self.assertFalse(distance < 1e-10)


class BuildSpottyMeshTestCase(ElisaTestCase):
    def generator_test_mesh(self, key, d):
        s = prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS[key],
                                  spots=testutils.SPOTS_META["primary"])
        s.star.discretization_factor = d
        system_container = testutils.prepare_single_system_container(s)
        system_container.build_mesh()

        self.assertTrue(len(system_container.star.spots) == 1)
        self.assertTrue(not is_empty(system_container.star.spots[0].points))

    def test_build_mesh(self):
        for key in testutils.SINGLE_SYSTEM_PARAMS.keys():
            self.generator_test_mesh(key=key, d=up.radians(5))

    @staticmethod
    def test_build_mesh_detached_with_overlapped_like_umbra():
        s = prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS['spherical'],
                                  spots=list(reversed(testutils.SPOTS_OVERLAPPED)))
        s.star.discretization_factor = up.radians(5)
        system_container = testutils.prepare_single_system_container(s)
        system_container.build_mesh()


class MeshUtilsTestCase(ElisaTestCase):
    def setUp(self):
        super(MeshUtilsTestCase, self).setUp()
        self.params_combination = [
            # solar model
            {
                "mass": 1.0,
                "t_eff": 5772 * u.K,
                "gravity_darkening": 0.32,
                # "polar_log_g": '4.43775 dex(cm / s2)',
                # "polar_log_g": 4.43775 * u.dex(u.cm/u.s**2),
                "polar_log_g": 4.43775,
                "gamma": 0.0,
                "inclination": 90.0 * u.deg,
                "rotation_period": 25.38 * u.d,
            },
        ]

        self._singles = self.prepare_systems()

    def prepare_systems(self):
        return [prepare_single_system(combo) for combo in self.params_combination]

    def test_surface_potential_from_polar_log_g(self):
        expected = np.round(np.array([-1.90691272573e+11]), -7)
        obtained = list()

        for ss in self._singles:
            obtained.append(model.surface_potential_from_polar_log_g(ss.star.polar_log_g, ss.star.mass))

        obtained = np.round(np.array(obtained), -7)
        assert_array_equal(expected, obtained)

    def test_potential(self):
        expected = np.round(np.array([-190761017680.0]), -7)
        obtained = list()
        radii = [695700000.0]

        for ii, ss in enumerate(self._singles):
            p_args = (ss.star.mass, ss.angular_velocity, 0.0)
            args = model.pre_calculate_for_potential_value(*p_args)
            obtained.append(model.potential(radii[ii], *args))

        obtained = np.round(np.array(obtained), -7)
        assert_array_equal(expected, obtained)

    def test_radial_potential_derivative(self):
        expected = np.round(np.array([274.200111657]), 3)  # this value should be always true for solar model since it
        # is surface gravity acceleration
        obtained = list()
        radii = [695700000.0]

        for ii, ss in enumerate(self._singles):
            p_args = (ss.star.mass, ss.angular_velocity, 0.0)
            args = (model.pre_calculate_for_potential_value(*p_args),)
            obtained.append(model.radial_potential_derivative(radii[ii], *args[0]))

        obtained = np.round(np.array(obtained), 3)
        assert_array_equal(expected, obtained)


@pytest.mark.parametrize('stage', ['center', 'first_ring'])
def test_single_star_spot_root_cannot_omit_spot(monkeypatch, stage):
    from elisa.single_system.surface import mesh as single_mesh
    single = testutils.prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS['spherical'],
                                          spots=testutils.SPOTS_META['primary'])
    container = testutils.prepare_single_system_container(single)
    original = single_mesh.calculate_radius
    spot = container.star.spots[0]
    target_theta = spot.latitude if stage == 'center' else spot.latitude + spot.discretization_factor
    injected = False

    def failed_radius(mass, angular_velocity, surface_potential, theta):
        nonlocal injected
        if theta == target_theta:
            injected = True
            return container.star.equatorial_radius * 2
        return original(mass, angular_velocity, surface_potential, theta)

    monkeypatch.setattr(single_mesh, 'calculate_radius', failed_radius)
    diagnostic = 'Center of spot' if stage == 'center' else 'First ring of spot'
    with pytest.raises(error.SpotError, match=diagnostic) as caught:
        single_mesh.mesh_spots(container)
    assert injected
    assert 'reasonable conditions' in str(caught.value)
    assert container.star.spots


def test_single_star_spot_surface_failure_preserves_cause(monkeypatch):
    from elisa.single_system.surface import mesh as single_mesh
    single = testutils.prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS['spherical'],
                                          spots=testutils.SPOTS_META['primary'])
    container = testutils.prepare_single_system_container(single)
    failure = error.MaxIterationError('spot surface failed')

    def exhausted(*args):
        raise failure

    monkeypatch.setattr(single_mesh, 'get_surface_points_radii', exhausted)
    with pytest.raises(error.SpotError) as caught:
        single_mesh.mesh_spots(container)
    assert caught.value.__cause__ is failure
    assert container.star.spots


@pytest.mark.parametrize('stage', ['center', 'first_ring'])
def test_single_spot_invalid_radius_source_keeps_spot_type_and_cause(monkeypatch, stage):
    from elisa.single_system import radius
    from elisa.single_system.surface import mesh as single_mesh
    single = testutils.prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS['spherical'],
                                          spots=testutils.SPOTS_META['primary'])
    container = testutils.prepare_single_system_container(single)
    original = radius.fsolve
    spot = container.star.spots[0]
    center_theta = spot.latitude
    first_ring_theta = spot.latitude + spot.discretization_factor
    # The radius solver receives potential coefficients, not the angle itself.
    center_coefficients = model.pre_calculate_for_potential_value(
        container.star.mass, container.angular_velocity, center_theta)
    first_ring_coefficients = model.pre_calculate_for_potential_value(
        container.star.mass, container.angular_velocity, first_ring_theta)
    assert not np.array_equal(center_coefficients, first_ring_coefficients)
    target_coefficients = center_coefficients if stage == 'center' else first_ring_coefficients
    injected = False

    def invalid_root(*args, **kwargs):
        nonlocal injected
        if np.array_equal(kwargs['args'][0], target_coefficients):
            injected = True
            return np.array([-1.]), {}, 1, 'wrong root'
        return original(*args, **kwargs)

    monkeypatch.setattr(radius, 'fsolve', invalid_root)
    with pytest.raises(error.SpotError, match='surface root') as caught:
        single_mesh.mesh_spots(container)
    assert injected
    assert isinstance(caught.value.__cause__, error.ElisaNumericError)
    assert 'Invalid value of radius' in str(caught.value.__cause__)
    assert container.star.spots
