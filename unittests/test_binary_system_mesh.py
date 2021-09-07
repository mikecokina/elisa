# keep it first
# due to stupid astropy units/constants implementation
from unittests import set_astropy_units

import numpy as np
from numpy.testing import assert_array_equal

from elisa import umpy as up, const, units as u, settings
from elisa.base.container import StarContainer
from elisa.binary_system import model
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.const import Position
from elisa.utils import is_empty, find_nearest_dist_3d
from unittests import utils as testutils
from unittests.utils import ElisaTestCase, prepare_binary_system

set_astropy_units()


class BuildMeshSpotsFreeTestCase(ElisaTestCase):
    @staticmethod
    def generator_test_mesh(key, d, length):
        s = prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key])
        s.primary.discretization_factor = d
        s.secondary.discretization_factor = d

        orbital_position_container = OrbitalPositionContainer.from_binary_system(
            binary_system=s,
            position=Position(*(0, 1.0, 0.0, 0.0, 0.0)),
        )
        orbital_position_container.build_mesh(components_distance=1.0)

        obtained_primary = np.round(orbital_position_container.primary.points, 4)
        obtained_secondary = np.round(orbital_position_container.secondary.points, 4)
        assert_array_equal([len(obtained_primary), len(obtained_secondary)], length)

    def test_build_mesh_detached_no_spot(self):
        self.generator_test_mesh(key="detached", d=up.radians(10), length=[418, 418])

    def test_build_mesh_overcontact_no_spot(self):
        self.generator_test_mesh(key="over-contact", d=up.radians(10), length=[377, 365])

    def test_build_mesh_overcontact_no_spot2(self):
        settings.configure(MESH_GENERATOR="improved_trapezoidal")
        self.generator_test_mesh(key="over-contact", d=up.radians(10), length=[377, 365])

    def test_build_mesh_semi_detached_no_spot(self):
        self.generator_test_mesh(key="semi-detached", d=up.radians(10), length=[418, 418])

    def test_mesh_for_duplicate_points_no_spot(self):
        for params in testutils.BINARY_SYSTEM_PARAMS.values():
            s = prepare_binary_system(params)
            # reducing runtime of the test
            s.primary.discretization_factor = up.radians(7)
            s.init()
            components_distance = s.orbit.orbital_motion(phase=0.0)[0][0]

            orbital_position_container = OrbitalPositionContainer(
                primary=StarContainer.from_properties_container(s.primary.to_properties_container()),
                secondary=StarContainer.from_properties_container(s.secondary.to_properties_container()),
                position=Position(*(0, 1.0, 0.0, 0.0, 0.0)),
                **s.properties_serializer()
            )

            orbital_position_container.build_mesh(components_distance=components_distance)

            for component in ['primary', 'secondary']:
                component_instance = getattr(orbital_position_container, component)
                distance = round(find_nearest_dist_3d(list(component_instance.points)), 10)

                if distance < 1e-10:
                    bad_points = []
                    for i, point in enumerate(component_instance.points):
                        for j, xx in enumerate(component_instance.points[i + 1:]):
                            dist = np.linalg.norm(point - xx)
                            if dist <= 1e-10:
                                print(f'Match: {point}, {i}, {j}')
                                bad_points.append(point)
                self.assertFalse(distance < 1e-10)

    def test_outliers(self):
        """
        checking if the spacing between vertices is beyond reasonable limits
        """
        for params in testutils.BINARY_SYSTEM_PARAMS.values():
            s = prepare_binary_system(params)
            # reducing runtime of the test
            s.primary.discretization_factor = up.radians(8)
            s.secondary.discretization_factor = up.radians(8)
            s.init()

            components_distance = s.orbit.orbital_motion(phase=0.0)[0][0]

            orbital_position_container = \
                OrbitalPositionContainer.from_binary_system(s, Position(*(0, components_distance, 0.0, 0.0, 0.0)))

            orbital_position_container.build_mesh(components_distance=components_distance, component='all')
            orbital_position_container.build_faces(components_distance=components_distance, component='all')

            primary = orbital_position_container.primary
            secondary = orbital_position_container.secondary
            points_p = primary.points[primary.faces]
            points_s = secondary.points[secondary.faces]

            distances_p = np.linalg.norm(np.diff(points_p, axis=1, prepend=points_p[:, 2:, :]), axis=2)
            distances_s = np.linalg.norm(np.diff(points_s, axis=1, prepend=points_s[:, 2:, :]), axis=2)

            p_triangle_size = primary.equivalent_radius * np.sin(s.primary.discretization_factor)
            s_triangle_size = secondary.equivalent_radius * np.sin(s.secondary.discretization_factor)
            self.assertGreater(3*p_triangle_size, distances_p.max())
            self.assertGreater(3*s_triangle_size, distances_s.max())


class BuildSpottyMeshTestCase(ElisaTestCase):
    def generator_test_mesh(self, key, d):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = d
        s.secondary.discretization_factor = d
        orbital_position_container = OrbitalPositionContainer(
            primary=StarContainer.from_properties_container(s.primary.to_properties_container()),
            secondary=StarContainer.from_properties_container(s.secondary.to_properties_container()),
            position=Position(*(0, 1.0, 0.0, 0.0, 0.0)),
            **s.properties_serializer()
        )
        orbital_position_container.build_mesh(components_distance=1.0)

        self.assertTrue(len(orbital_position_container.primary.spots) == 1
                        and len(orbital_position_container.secondary.spots) == 1)
        self.assertTrue(not is_empty(orbital_position_container.primary.spots[0].points))
        self.assertTrue(not is_empty(orbital_position_container.secondary.spots[0].points))

    def test_build_mesh_detached(self):
        self.generator_test_mesh(key="detached", d=up.radians(10))

    def test_build_mesh_overcontact(self):
        self.generator_test_mesh(key="over-contact", d=up.radians(10))

    def test_build_mesh_semi_detached(self):
        self.generator_test_mesh(key="semi-detached", d=up.radians(10))

    def test_build_mesh_detached_with_overlapped_spots(self):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS['detached'],
                                            spots_primary=testutils.SPOTS_OVERLAPPED)
        s.primary.discretization_factor = up.radians(5)
        s.secondary.discretization_factor = up.radians(5)
        orbital_position_container = OrbitalPositionContainer(
            primary=StarContainer.from_properties_container(s.primary.to_properties_container()),
            secondary=StarContainer.from_properties_container(s.secondary.to_properties_container()),
            position=Position(*(0, 1.0, 0.0, 0.0, 0.0)),
            **s.properties_serializer()
        )
        with self.assertRaises(Exception) as context:
            orbital_position_container.build_mesh(components_distance=1.0)
        self.assertTrue("Please, specify spots wisely" in str(context.exception))

    @staticmethod
    def test_build_mesh_detached_with_overlapped_like_umbra():
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS['detached'],
                                            spots_primary=list(reversed(testutils.SPOTS_OVERLAPPED)))
        s.primary.discretization_factor = up.radians(5)
        s.secondary.discretization_factor = up.radians(5)
        orbital_position_container = OrbitalPositionContainer(
            primary=StarContainer.from_properties_container(s.primary.to_properties_container()),
            secondary=StarContainer.from_properties_container(s.secondary.to_properties_container()),
            position=Position(*(0, 1.0, 0.0, 0.0, 0.0)),
            **s.properties_serializer()
        )
        orbital_position_container.build_mesh(components_distance=1.0)

    def test_make_sure_spots_are_not_overwriten_in_star_instance(self):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS["detached"],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"]
                                            )
        s.primary.discretization_factor = up.radians(10)
        s.secondary.discretization_factor = up.radians(10)

        orbital_position_container = OrbitalPositionContainer(
            primary=StarContainer.from_properties_container(s.primary.to_properties_container()),
            secondary=StarContainer.from_properties_container(s.secondary.to_properties_container()),
            position=Position(*(0, 1.0, 0.0, 0.0, 0.0)),
            **s.properties_serializer()
        )
        orbital_position_container.build_mesh(components_distance=1.0)
        self.assertTrue(is_empty(s.primary.spots[0].points))
        self.assertTrue(is_empty(s.secondary.spots[0].points))

        self.assertTrue(is_empty(s.primary.spots[0].faces))
        self.assertTrue(is_empty(s.secondary.spots[0].faces))

    def test_spotty_mesh_for_duplicate_points(self):
        for params in testutils.BINARY_SYSTEM_PARAMS.values():
            s = testutils.prepare_binary_system(params,
                                                spots_primary=testutils.SPOTS_META["primary"],
                                                spots_secondary=testutils.SPOTS_META["secondary"]
                                                )
            components_distance = s.orbit.orbital_motion(phase=0.0)[0][0]

            orbital_position_container = OrbitalPositionContainer(
                primary=StarContainer.from_properties_container(s.primary.to_properties_container()),
                secondary=StarContainer.from_properties_container(s.secondary.to_properties_container()),
                position=Position(*(0, 1.0, 0.0, 0.0, 0.0)),
                **s.properties_serializer()
            )

            orbital_position_container.build_mesh(components_distance=components_distance)

            for component in ['primary', 'secondary']:
                component_instance = getattr(orbital_position_container, component)
                distance = round(find_nearest_dist_3d(list(component_instance.points)), 10)

                if distance < 1e-10:
                    bad_points = []
                    for i, point in enumerate(component_instance.points):
                        for j, xx in enumerate(component_instance.points[i + 1:]):
                            dist = np.linalg.norm(point - xx)
                            if dist <= 1e-10:
                                print(f'Match: {point}, {i}, {j}')
                                bad_points.append(point)

                self.assertFalse(distance < 1e-10)

                spot_distance = round(find_nearest_dist_3d(list(component_instance.spots[0].points)), 10)

                if spot_distance < 1e-10:
                    bad_points = []
                    for i, point in enumerate(component_instance.spots[0].points):
                        for j, xx in enumerate(component_instance.spots[0].points[i + 1:]):
                            dist = np.linalg.norm(point - xx)
                            if dist <= 1e-10:
                                print(f'Match: {point}, {i}, {j}')
                                bad_points.append(point)

                self.assertFalse(spot_distance < 1e-10)

    def test_outliers(self):
        """
        checking if the spacing between vertices is beyond reasonable limits
        """
        for params in testutils.BINARY_SYSTEM_PARAMS.values():
            s = prepare_binary_system(params, spots_primary=testutils.SPOTS_META["primary"],
                                      spots_secondary=testutils.SPOTS_META["secondary"])
            # reducing runtime of the test
            s.primary.discretization_factor = up.radians(8)
            s.secondary.discretization_factor = up.radians(8)
            s.init()

            components_distance = s.orbit.orbital_motion(phase=0.0)[0][0]

            orbital_position_container = \
                OrbitalPositionContainer.from_binary_system(s, Position(*(0, components_distance, 0.0, 0.0, 0.0)))

            orbital_position_container.build_mesh(components_distance=components_distance, component='all')
            orbital_position_container.build_faces(components_distance=components_distance, component='all')
            # orbital_position_container.flat_it()

            primary = orbital_position_container.primary
            secondary = orbital_position_container.secondary
            points_p = primary.get_flatten_parameter("points")[primary.get_flatten_parameter("faces")]
            points_s = secondary.get_flatten_parameter("points")[secondary.get_flatten_parameter("faces")]

            distances_p = np.linalg.norm(np.diff(points_p, axis=1, prepend=points_p[:, 2:, :]), axis=2)
            distances_s = np.linalg.norm(np.diff(points_s, axis=1, prepend=points_s[:, 2:, :]), axis=2)

            p_triangle_size = primary.equivalent_radius * np.sin(s.primary.discretization_factor)
            s_triangle_size = secondary.equivalent_radius * np.sin(s.secondary.discretization_factor)
            self.assertGreater(4*p_triangle_size, distances_p.max())
            self.assertGreater(4*s_triangle_size, distances_s.max())


class MeshUtilsTestCase(ElisaTestCase):
    def setUp(self):
        super(MeshUtilsTestCase, self).setUp()
        self.params_combination = [
            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
             "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
             "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.0, "inclination": const.HALF_PI * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6,
             },
            # compact spherical components on circular orbit

            {"primary_mass": 2.0, "secondary_mass": 1.0,
             "primary_surface_potential": 4.8, "secondary_surface_potential": 4.0,
             "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
             "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
             "eccentricity": 0.3, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
             "phase_shift": 0.0,
             "primary_t_eff": 5000, "secondary_t_eff": 5000,
             "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
             "primary_albedo": 0.6, "secondary_albedo": 0.6
             }  # close tidally deformed components with asynchronous rotation on eccentric orbit
        ]

        self._binaries = self.prepare_systems()

    def prepare_systems(self):
        return [prepare_binary_system(combo) for combo in self.params_combination]

    def test_primary_potential_derivative_x(self):
        d, x = 1.1, 0.13
        expected = np.round(np.array([-58.8584146731, -58.6146646731]), 4)
        obtained = list()

        for bs in self._binaries:
            args = (bs.primary.synchronicity, bs.mass_ratio, d)
            obtained.append(model.primary_potential_derivative_x(x, *args))
        obtained = np.round(obtained, 4)
        assert_array_equal(expected, obtained)

    def test_secondary_potential_derivative_x(self):
        d, x = 1.1, 0.13
        expected = np.round(np.array([-59.268745, -59.908945]), 4)
        obtained = list()

        for bs in self._binaries:
            args = (bs.secondary.synchronicity, bs.mass_ratio, d)
            obtained.append(model.secondary_potential_derivative_x(x, *args))
        obtained = np.round(obtained, 4)
        assert_array_equal(expected, obtained)

    def test_pre_calculate_for_potential_value_primary(self):
        # single
        distance, phi, theta = 1.1, const.HALF_PI, const.HALF_PI / 2.0
        args = (distance, phi, theta)

        obtained = list()
        expected = np.round([[1.21, 0., 0., 0.375], [1.21, 0., 0., 0.8438]], 3)

        for bs in self._binaries:
            argss = (bs.primary.synchronicity, bs.mass_ratio) + args
            obtained.append(model.pre_calculate_for_potential_value_primary(*argss))

        obtained = np.round(obtained, 3)
        assert_array_equal(expected, obtained)

    def test_pre_calculate_for_potential_value_secondary(self):
        # single
        distance, phi, theta = 1.1, const.HALF_PI, const.HALF_PI / 2.0
        args = (distance, phi, theta)

        obtained = list()
        expected = np.round([[1.21, 0., 0., 0.375, 0.25], [1.21, 0., 0., 0.8438, 0.25]], 3)

        for bs in self._binaries:
            argss = (bs.primary.synchronicity, bs.mass_ratio) + args
            obtained.append(model.pre_calculate_for_potential_value_secondary(*argss))

        obtained = np.round(obtained, 3)
        assert_array_equal(expected, obtained)
