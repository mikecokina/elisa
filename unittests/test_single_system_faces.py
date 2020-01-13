import numpy as np

from elisa import umpy as up
from unittests import utils as testutils
from unittests.utils import ElisaTestCase
from elisa.utils import is_empty


class BuildFacesSpotsFreeTestCase(ElisaTestCase):
    @staticmethod
    def build_system(key, d):
        s = testutils.prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS[key])
        s.star.discretization_factor = d

        position_container = testutils.prepare_single_system_container(s)
        position_container.build_mesh()
        position_container.build_faces()
        return position_container

    def test_closed_surface_spherical(self):
        position_container = self.build_system('spherical', up.radians(10))
        self.assertTrue(testutils.surface_closed(faces=position_container.star.faces,
                                                 points=position_container.star.points))

    def test_closed_surface_squashed(self):
        position_container = self.build_system('squashed', up.radians(10))
        self.assertTrue(testutils.surface_closed(faces=position_container.star.faces,
                                                 points=position_container.star.points))


class BuildSpottyFacesTestCase(ElisaTestCase):
    @staticmethod
    def build_system(key, d):
        s = testutils.prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS[key],
                                            spots=testutils.SPOTS_META["primary"])
        s.star.discretization_factor = d

        position_container = testutils.prepare_single_system_container(s)
        position_container.build_mesh()
        position_container.build_faces()

        instance = getattr(position_container, 'star')
        points = instance.points
        faces = instance.faces
        if isinstance(instance.spots, (dict,)):
            for idx, spot in instance.spots.items():
                faces = up.concatenate((faces, spot.faces + len(points)), axis=0)
                points = up.concatenate((points, spot.points), axis=0)
        setattr(instance, 'points', points)
        setattr(instance, 'faces', faces)
        return position_container

    def test_closed_surface_spherical(self):
        position_container = self.build_system('spherical', up.radians(10))
        self.assertTrue(testutils.surface_closed(faces=position_container.star.faces,
                                                 points=position_container.star.points))

    def test_closed_surface_squashed(self):
        position_container = self.build_system('squashed', up.radians(10))
        self.assertTrue(testutils.surface_closed(faces=position_container.star.faces,
                                                 points=position_container.star.points))


class BuildSpottyFacesOrientationTestCase(ElisaTestCase):
    def generator_test_face_orientaion(self, key, kind):
        s = testutils.prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS[key],
                                            spots=testutils.SPOTS_META["primary"])
        s.star.discretization_factor = up.radians(10)
        position_container = testutils.prepare_single_system_container(s)
        position_container.build_mesh()
        position_container.build_faces()
        position_container.build_surface_areas()
        position_container.build_faces_orientation()

        if kind == 'present':
            self.assertTrue(not is_empty(position_container.star.normals))

            self.assertTrue(not is_empty(position_container.star.spots[0].normals))

        if kind == 'direction':
            o = position_container
            t = 1e-5 * np.max(position_container.star.points)
            _assert = self.assertTrue
            # x axis
            # TODO: finish this for all axis and spots
            _assert(np.all(o.star.normals[o.star.points[o.star.faces][:, 0, 0] > t][:, 0] > 0))

    def test_if_normals_present_spherical(self):
        self.generator_test_face_orientaion('spherical', 'present')

    def test_if_normals_present_squashed(self):
        self.generator_test_face_orientaion('squashed', 'present')

    def test_normals_direction_spherical(self):
        self.generator_test_face_orientaion('spherical', 'direction')

    # def test_normals_direction_squashed(self):
    #     self.generator_test_face_orientaion('squashed', 'direction')
