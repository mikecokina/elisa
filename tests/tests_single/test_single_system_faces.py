from __future__ import annotations

from typing import TYPE_CHECKING

# The tests package selects Astropy constants before test-module imports.
import numpy as np

from elisa import umpy as up
from elisa.utils import is_empty
from tests import set_astropy_units
from tests import utils as testutils
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    import elisa.single_system.container


set_astropy_units()


class BuildFacesSpotsFreeTestCase(ElisaTestCase):
    @staticmethod
    def build_system(key: str, d: np.float64) -> elisa.single_system.container.SinglePositionContainer:
        s = testutils.prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS[key])
        s.star.discretization_factor = d

        position_container = testutils.prepare_single_system_container(s)
        position_container.build_mesh()
        position_container.build_faces()
        return position_container

    def test_closed_surface_spherical(self) -> None:
        position_container = self.build_system("spherical", up.radians(10))
        assert testutils.surface_closed(faces=position_container.star.faces, points=position_container.star.points)

    def test_closed_surface_squashed(self) -> None:
        position_container = self.build_system("squashed", up.radians(10))
        assert testutils.surface_closed(faces=position_container.star.faces, points=position_container.star.points)


class BuildSpottyFacesTestCase(ElisaTestCase):
    @staticmethod
    def build_system(key: str, d: np.float64) -> elisa.single_system.container.SinglePositionContainer:
        s = testutils.prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS[key], spots=testutils.SPOTS_META["primary"])
        s.star.discretization_factor = d
        s.init()

        position_container = testutils.prepare_single_system_container(s)
        position_container.build_mesh()
        position_container.build_faces()

        instance = position_container.star
        points = instance.points
        faces = instance.faces
        if isinstance(instance.spots, (dict,)):
            for spot in instance.spots.values():
                faces = up.concatenate((faces, spot.faces + len(points)), axis=0)
                points = up.concatenate((points, spot.points), axis=0)
        instance.points = points
        instance.faces = faces
        return position_container

    def test_closed_surface_spherical(self) -> None:
        position_container = self.build_system("spherical", up.radians(10))
        assert testutils.surface_closed(faces=position_container.star.faces, points=position_container.star.points)

    def test_closed_surface_squashed(self) -> None:
        position_container = self.build_system("squashed", up.radians(10))
        assert testutils.surface_closed(faces=position_container.star.faces, points=position_container.star.points)


class BuildSpottyFacesOrientationTestCase(ElisaTestCase):
    def generator_test_face_orientaion(self, key: str, kind: str) -> None:
        s = testutils.prepare_single_system(
            testutils.SINGLE_SYSTEM_PARAMS[key],
            spots=testutils.SPOTS_META["primary"],
        )
        s.star.discretization_factor = up.radians(7)
        s.init()
        position_container = testutils.prepare_single_system_container(s)
        position_container.build_mesh()
        position_container.build_faces()
        position_container.build_surface_areas()
        position_container.build_faces_orientation()

        if kind == "present":
            assert not is_empty(position_container.star.normals)

            assert not is_empty(position_container.star.spots[0].normals)

        _assert = self.assertTrue
        if kind == "direction":
            o = position_container

            face_points = o.star.points[o.star.faces]
            spot_face_points = o.star.spots[0].points[o.star.spots[0].faces]

            # x axis
            all_positive = (face_points[:, :, 0] >= 0).all(axis=1)
            _assert(np.all(o.star.normals[all_positive][:, 0] > 0))
            all_negative = (face_points[:, :, 0] <= 0).all(axis=1)
            _assert(np.all(o.star.normals[all_negative][:, 0] < 0))

            all_positive = (spot_face_points[:, :, 0] >= 0).all(axis=1)
            _assert(np.all(o.star.spots[0].normals[all_positive][:, 0] > 0))
            all_negative = (spot_face_points[:, :, 0] <= 0).all(axis=1)
            _assert(np.all(o.star.spots[0].normals[all_negative][:, 0] < 0))

            # y axis
            all_positive = (face_points[:, :, 1] >= 0).all(axis=1)
            _assert(np.all(o.star.normals[all_positive][:, 1] > 0))
            all_negative = (face_points[:, :, 1] <= 0).all(axis=1)
            _assert(np.all(o.star.normals[all_negative][:, 1] < 0))

            all_positive = (spot_face_points[:, :, 1] >= 0).all(axis=1)
            _assert(np.all(o.star.spots[0].normals[all_positive][:, 1] > 0))
            all_negative = (spot_face_points[:, :, 1] <= 0).all(axis=1)
            _assert(np.all(o.star.spots[0].normals[all_negative][:, 1] < 0))

            # z axis
            all_positive = (face_points[:, :, 2] >= 0).all(axis=1)
            _assert(np.all(o.star.normals[all_positive][:, 2] > 0))
            all_negative = (face_points[:, :, 2] <= 0).all(axis=1)
            _assert(np.all(o.star.normals[all_negative][:, 2] < 0))

            all_positive = (spot_face_points[:, :, 2] >= 0).all(axis=1)
            _assert(np.all(o.star.spots[0].normals[all_positive][:, 2] > 0))
            all_negative = (spot_face_points[:, :, 2] <= 0).all(axis=1)
            _assert(np.all(o.star.spots[0].normals[all_negative][:, 2] < 0))

        if kind == "size":
            o = position_container

            normals_size = np.linalg.norm(o.star.normals, axis=1)
            _assert((np.round(normals_size, 5) == 1).all())

            spot_normals_size = np.linalg.norm(o.star.spots[0].normals, axis=1)
            _assert((np.round(spot_normals_size, 5) == 1).all())

    def test_if_normals_present_spherical(self) -> None:
        self.generator_test_face_orientaion("spherical", "present")

    def test_if_normals_present_squashed(self) -> None:
        self.generator_test_face_orientaion("squashed", "present")

    def test_normals_direction_spherical(self) -> None:
        self.generator_test_face_orientaion("spherical", "direction")

    def test_normals_direction_squashed(self) -> None:
        self.generator_test_face_orientaion("squashed", "direction")

    def test_normals_size_spherical(self) -> None:
        self.generator_test_face_orientaion("spherical", "size")

    def test_normals_size_squashed(self) -> None:
        self.generator_test_face_orientaion("squashed", "size")
