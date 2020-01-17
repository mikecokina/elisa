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
        return orbital_position_container

    @staticmethod
    def generator_test_faces(key, d, length):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = d
        s.init()
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)

        assert_array_equal([len(orbital_position_container.primary.faces),
                            len(orbital_position_container.secondary.faces),
                            len(orbital_position_container.primary.spots[0].faces),
                            len(orbital_position_container.secondary.spots[0].faces)], length)

    def test_build_faces_detached(self):
        self.generator_test_faces('detached', up.radians(10), [785, 186, 97, 6])

    def test_build_faces_over_contact(self):
        self.generator_test_faces('over-contact', up.radians(10), [751, 374, 97, 24])

    def test_build_faces_semi_detached(self):
        self.generator_test_faces('semi-detached', up.radians(10), [785, 400, 97, 24])

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
        s.init()
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
            self.assertTrue(np.all(up.less(orbital_position_container.primary.areas, less)))
            self.assertTrue(np.all(up.less(orbital_position_container.secondary.areas, less)))

            self.assertTrue(np.all(up.less(orbital_position_container.primary.spots[0].areas, less)))
            self.assertTrue(np.all(up.less(orbital_position_container.secondary.spots[0].areas, less)))

    def test_build_surface_areas_detached(self):
        self.generator_test_surface_areas('detached', up.radians(10), kind="contain")

    def test_build_surface_areas_over_contact(self):
        self.generator_test_surface_areas('over-contact', up.radians(10), kind="contain")

    def test_build_surface_areas_semi_detached(self):
        self.generator_test_surface_areas('semi-detached', up.radians(10), kind="contain")

    def test_build_surface_areas_detached_size(self):
        self.generator_test_surface_areas('detached', up.radians(10), kind="size", less=5e-6)

    def test_build_surface_areas_over_contact_size(self):
        self.generator_test_surface_areas('over-contact', up.radians(10), kind="size", less=8e-3)

    def test_build_surface_areas_semi_detached_size(self):
        self.generator_test_surface_areas('semi-detached', up.radians(10), kind="size", less=8e-3)


class BuildSpottyFacesOrientationTestCase(ElisaTestCase):
    def generator_test_face_orientaion(self, key, kind):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = up.radians(7)
        s.init()
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

        _assert = self.assertTrue
        if kind == 'direction':
            o = orbital_position_container

            for component in ['primary', 'secondary']:
                star = getattr(o, component)

                face_points = star.points[star.faces]
                spot_face_points = star.spots[0].points[star.spots[0].faces]

                face_points[:, :, 0] = face_points[:, :, 0] - 1 if component == 'secondary' else face_points[:, :, 0]
                spot_face_points[:, :, 0] = spot_face_points[:, :, 0] - 1 if component == 'secondary' else \
                    spot_face_points[:, :, 0]

                # x axis
                all_positive = (face_points[:, :, 0] > 0).all(axis=1)
                _assert(np.all(star.normals[all_positive][:, 0] > 0))
                all_negative = (face_points[:, :, 0] < 0).all(axis=1)
                _assert(np.all(star.normals[all_negative][:, 0] < 0))

                all_positive = (spot_face_points[:, :, 0] > 0).all(axis=1)
                _assert(np.all(star.spots[0].normals[all_positive][:, 0] > 0))
                all_negative = (spot_face_points[:, :, 0] <= 0).all(axis=1)
                _assert(np.all(star.spots[0].normals[all_negative][:, 0] < 0))

                # y axis
                all_positive = (face_points[:, :, 1] > 0).all(axis=1)
                _assert(np.all(star.normals[all_positive][:, 1] > 0))
                all_negative = (face_points[:, :, 1] < 0).all(axis=1)
                _assert(np.all(star.normals[all_negative][:, 1] < 0))

                all_positive = (spot_face_points[:, :, 1] > 0).all(axis=1)
                _assert(np.all(star.spots[0].normals[all_positive][:, 1] > 0))
                all_negative = (spot_face_points[:, :, 1] < 0).all(axis=1)
                _assert(np.all(star.spots[0].normals[all_negative][:, 1] < 0))

                # z axis
                all_positive = (face_points[:, :, 2] > 0).all(axis=1)
                _assert(np.all(star.normals[all_positive][:, 2] > 0))
                all_negative = (face_points[:, :, 2] < 0).all(axis=1)
                _assert(np.all(star.normals[all_negative][:, 2] < 0))

                all_positive = (spot_face_points[:, :, 2] > 0).all(axis=1)
                _assert(np.all(star.spots[0].normals[all_positive][:, 2] > 0))
                all_negative = (spot_face_points[:, :, 2] < 0).all(axis=1)
                _assert(np.all(star.spots[0].normals[all_negative][:, 2] < 0))

        if kind == 'size':
            o = orbital_position_container

            for component in ['primary', 'secondary']:
                star = getattr(o, component)

                normals_size = np.linalg.norm(star.normals, axis=1)
                _assert((np.round(normals_size, 5) == 1).all())

                spot_normals_size = np.linalg.norm(star.spots[0].normals, axis=1)
                _assert((np.round(spot_normals_size, 5) == 1).all())

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

    def test_normals_size_detached(self):
        self.generator_test_face_orientaion('detached', 'size')

    def test_normals_size_semi_detached(self):
        self.generator_test_face_orientaion('semi-detached', 'size')

    def test_normals_size_overcontact(self):
        self.generator_test_face_orientaion('over-contact', 'size')
