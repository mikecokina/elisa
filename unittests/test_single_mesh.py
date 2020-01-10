import numpy as np

from numpy.testing import assert_array_equal

from unittests import utils as testutils
from unittests.utils import ElisaTestCase, prepare_single_system

from elisa.base.container import StarContainer
from elisa.single_system.container import SystemContainer
from elisa.const import SinglePosition
from elisa.utils import is_empty, find_nearest_dist_3d
from elisa import umpy as up, units


class BuildMeshSpotsFreeTestCase(ElisaTestCase):
    def test_mesh_for_duplicate_points(self):
        for params in testutils.SINGLE_SYSTEM_PARAMS.values():
            s = prepare_single_system(params)
            # reducing runtime of the test
            s.star.discretization_factor = up.radians(7)
            s.init()

            system_container = SystemContainer(
                star=StarContainer.from_properties_container(s.star.to_properties_container()),
                position=SinglePosition(*(0, 0.0, 0.0)),
                **s.properties_serializer()
            )

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
        system_container = SystemContainer(
            star=StarContainer.from_properties_container(s.star.to_properties_container()),
            position=SinglePosition(*(0, 0.0, 0.0)),
            **s.properties_serializer()
        )

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
        system_container = SystemContainer(
            star=StarContainer.from_properties_container(s.star.to_properties_container()),
            position=SinglePosition(*(0, 0.0, 0.0)),
            **s.properties_serializer()
        )
        system_container.build_mesh()


class MeshUtilsTestCase(ElisaTestCase):
    def setUp(self):
        self.params_combination = [
            # solar model
            {
                "mass": 1.0,
                "t_eff": 5772 * units.K,
                "gravity_darkening": 0.32,
                "polar_log_g": 4.43775,
                "gamma": 0.0,
                "inclination": 90.0 * units.deg,
                "rotation_period": 25.38 * units.d,
            },
        ]

        self._binaries = self.prepare_systems()

    def prepare_systems(self):
        return [prepare_single_system(combo) for combo in self.params_combination]

    # TODO: continue with tests on surface generating functions
