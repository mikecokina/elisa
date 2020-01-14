import numpy as np
from numpy.testing import assert_array_equal

from elisa import umpy as up
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.utils import is_empty
from unittests import utils as testutils
from unittests.utils import ElisaTestCase


class BuildFacesSpotsFreeTestCase(ElisaTestCase):
    @staticmethod
    def build_system(key, d):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key])
        s.primary.discretization_factor = d
        s.secondary.discretization_factor = d

        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        return orbital_position_container

    def generator_test_faces(self, key, d, length):
        orbital_position_container = self.build_system(key, d)

        assert_array_equal([len(orbital_position_container.primary.faces),
                            len(orbital_position_container.secondary.faces)], length)

    def test_build_faces_detached(self):
        self.generator_test_faces('detached', up.radians(10), [848, 848])

    def test_build_faces_over_contact(self):
        self.generator_test_faces('over-contact', up.radians(10), [812, 784])

    def test_build_faces_semi_detached(self):
        self.generator_test_faces('semi-detached', up.radians(10), [848, 848])

    def test_closed_surface_detached(self):
        orbital_position_container = self.build_system('detached', up.radians(10))
        self.assertTrue(testutils.surface_closed(faces=orbital_position_container.primary.faces,
                                                 points=orbital_position_container.primary.points))

    def test_closed_surface_semi_detached(self):
        orbital_position_container = self.build_system('semi-detached', up.radians(10))
        self.assertTrue(testutils.surface_closed(faces=orbital_position_container.primary.faces,
                                                 points=orbital_position_container.primary.points))

    def test_closed_surface_over_contact(self):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS['over-contact'])
        s.primary.discretization_factor = up.radians(10)
        s.init()

        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)

        points = np.concatenate((orbital_position_container.primary.points,
                                 orbital_position_container.secondary.points))
        faces = np.concatenate((orbital_position_container.primary.faces,
                                orbital_position_container.secondary.faces +
                                np.shape(orbital_position_container.primary.points)[0]))
        self.assertTrue(testutils.surface_closed(faces=faces, points=points))


class BuildSpottyFacesTestCase(ElisaTestCase):
    @staticmethod
    def build_system(key, d):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = d
        s.secondary.discretization_factor = d

        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        for component in ['primary', 'secondary']:
            instance = getattr(orbital_position_container, component)
            points = instance.points
            faces = instance.faces
            if isinstance(instance.spots, (dict,)):
                for idx, spot in instance.spots.items():
                    faces = up.concatenate((faces, spot.faces + len(points)), axis=0)
                    points = up.concatenate((points, spot.points), axis=0)
            setattr(instance, 'points', points)
            setattr(instance, 'faces', faces)
        return orbital_position_container

    @staticmethod
    def generator_test_faces(key, d, length):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = d
        s.secondary.discretization_factor = d
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)

        assert_array_equal([len(orbital_position_container.primary.faces),
                            len(orbital_position_container.secondary.faces),
                            len(orbital_position_container.primary.spots[0].faces),
                            len(orbital_position_container.secondary.spots[0].faces)], length)

    def test_build_faces_detached(self):
        self.generator_test_faces('detached', up.radians(10), [849, 858, 1049, 618])

    def test_build_faces_over_contact(self):
        self.generator_test_faces('over-contact', up.radians(10), [821, 804, 1049, 618])

    def test_build_faces_semi_detached(self):
        self.generator_test_faces('semi-detached', up.radians(10), [849, 858, 1049, 618])

    def test_closed_surface_detached(self):
        orbital_position_container = self.build_system('detached', up.radians(10))
        self.assertTrue(testutils.surface_closed(faces=orbital_position_container.primary.faces,
                                                 points=orbital_position_container.primary.points))

    def test_closed_surface_semi_detached(self):
        orbital_position_container = self.build_system('semi-detached', up.radians(10))
        self.assertTrue(testutils.surface_closed(faces=orbital_position_container.primary.faces,
                                                 points=orbital_position_container.primary.points))

    def test_closed_surface_over_contact(self):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS['over-contact'],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = up.radians(7)
        s.init()

        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        for component in ['primary', 'secondary']:
            instance = getattr(orbital_position_container, component)
            points = instance.points
            faces = instance.faces
            if isinstance(instance.spots, (dict,)):
                for idx, spot in instance.spots.items():
                    faces = up.concatenate((faces, spot.faces + len(points)), axis=0)
                    points = up.concatenate((points, spot.points), axis=0)
            setattr(instance, 'points', points)
            setattr(instance, 'faces', faces)

        points = np.concatenate((orbital_position_container.primary.points,
                                 orbital_position_container.secondary.points))
        faces = np.concatenate((orbital_position_container.primary.faces,
                                orbital_position_container.secondary.faces +
                                np.shape(orbital_position_container.primary.points)[0]))
        self.assertTrue(testutils.surface_closed(faces=faces, points=points))


class BuildSurfaceAreasTestCase(ElisaTestCase):
    def generator_test_surface_areas(self, key, d, kind, less=None):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = d
        s.secondary.discretization_factor = d
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        orbital_position_container.build_surface_areas()

        if kind == "contain":
            self.assertTrue(not is_empty(orbital_position_container.primary.areas))
            self.assertTrue(not is_empty(orbital_position_container.secondary.areas))

            self.assertTrue(not is_empty(orbital_position_container.primary.spots[0].areas))
            self.assertTrue(not is_empty(orbital_position_container.secondary.spots[0].areas))

        if kind == "size":
            self.assertTrue(np.all(up.less(orbital_position_container.primary.areas, less[0])))
            self.assertTrue(np.all(up.less(orbital_position_container.secondary.areas, less[1])))

            self.assertTrue(np.all(up.less(orbital_position_container.primary.spots[0].areas, less[2])))
            self.assertTrue(np.all(up.less(orbital_position_container.secondary.spots[0].areas, less[3])))

    def test_build_surface_areas_detached(self):
        self.generator_test_surface_areas('detached', up.radians(10), kind="contain")

    def test_build_surface_areas_over_contact(self):
        self.generator_test_surface_areas('over-contact', up.radians(10), kind="contain")

    def test_build_surface_areas_semi_detached(self):
        self.generator_test_surface_areas('semi-detached', up.radians(10), kind="contain")

    def test_build_surface_areas_detached_size(self):
        self.generator_test_surface_areas('detached', up.radians(10), kind="size", less=[2e-6, 5e-7, 2e-7, 4e-8])

    def test_build_surface_areas_over_contact_size(self):
        self.generator_test_surface_areas('over-contact', up.radians(10), kind="size", less=[6e-3, 3e-3, 3e-4, 2e-4])

    def test_build_surface_areas_semi_detached_size(self):
        self.generator_test_surface_areas('semi-detached', up.radians(10), kind="size", less=[6e-3, 3e-3, 3e-4, 2e-4])


class BuildSpottyFacesOrientationTestCase(ElisaTestCase):
    def generator_test_face_orientaion(self, key, kind):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = up.radians(10)
        s.secondary.discretization_factor = up.radians(10)
        orbital_position_container: OrbitalPositionContainer = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        orbital_position_container.build_surface_areas()
        orbital_position_container.build_faces_orientation(components_distance=1.0)

        if kind == 'present':
            self.assertTrue(not is_empty(orbital_position_container.primary.normals))
            self.assertTrue(not is_empty(orbital_position_container.secondary.normals))

            self.assertTrue(not is_empty(orbital_position_container.primary.spots[0].normals))
            self.assertTrue(not is_empty(orbital_position_container.secondary.spots[0].normals))

        if kind == 'direction':
            o = orbital_position_container
            t = 1e-5
            _assert = self.assertTrue
            average_point_vector_primary = np.mean(o.primary.points[o.primary.faces], axis=1)
            average_point_vector_secondary = np.mean(o.secondary.points[o.secondary.faces], axis=1)
            # TODO: finish this for all axis and spots
            _assert(np.all(o.primary.normals[average_point_vector_primary[:, 0] > t][:, 0] > 0))
            _assert(np.all(o.secondary.normals[average_point_vector_secondary[:, 0] < t][:, 0] < 0))

            # _assert(np.all(o.primary.normals[average_point_vector_primary[:, 0] < t][:, 0] < 0))
            # _assert(np.all(o.secondary.normals[average_point_vector_secondary[:, 0] > t][:, 0] < 0))

    def test_if_normals_present_detached(self):
        self.generator_test_face_orientaion('detached', 'present')

    def test_if_normals_present_semi_detached(self):
        self.generator_test_face_orientaion('semi-detached', 'present')

    def test_if_normals_present_overcontact(self):
        self.generator_test_face_orientaion('over-contact', 'present')

    def test_normals_direction_detached(self):
        self.generator_test_face_orientaion('detached', 'direction')

    def test_normals_direction_semi_detached(self):
        self.generator_test_face_orientaion('semi-detached', 'direction')

    def test_normals_direction_overcontact(self):
        self.generator_test_face_orientaion('over-contact', 'direction')
