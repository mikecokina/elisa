from __future__ import annotations

from typing import TYPE_CHECKING

# The tests package selects Astropy constants before test-module imports.
import numpy as np
from numpy.testing import assert_array_equal

from elisa import settings
from elisa import umpy as up
from elisa import units as u
from elisa.utils import is_empty
from tests import set_astropy_units
from tests import utils as testutils
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    from elisa.binary_system.container import OrbitalPositionContainer


set_astropy_units()


class BuildFacesSpotsFreeTestCase(ElisaTestCase):
    @staticmethod
    def build_system(key: str) -> OrbitalPositionContainer:
        params = testutils.BINARY_SYSTEM_PARAMS[key].copy()
        params.update({"primary_discretization_factor": 10})
        s = testutils.prepare_binary_system(params)
        s.secondary.discretization_factor = 10 * u.deg
        s.init()

        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        return orbital_position_container

    def generator_test_faces(self, key: str, length: list[int]) -> None:
        orbital_position_container = self.build_system(key)
        assert_array_equal(
            [len(orbital_position_container.primary.faces), len(orbital_position_container.secondary.faces)],
            length,
        )

    def test_build_faces_detached(self) -> None:
        self.generator_test_faces("detached", [832, 832])

    def test_build_faces_over_contact(self) -> None:
        self.generator_test_faces("over-contact", [736, 708])

    def test_build_faces_over_contact2(self) -> None:
        settings.configure(MESH_GENERATOR="improved_trapezoidal")
        self.generator_test_faces("over-contact", [736, 712])

    def test_build_faces_semi_detached(self) -> None:
        self.generator_test_faces("semi-detached", [832, 832])

    def test_build_faces_semi_detached2(self) -> None:
        settings.configure(MESH_GENERATOR="improved_trapezoidal")
        self.generator_test_faces("semi-detached", [888, 888])

    def test_closed_surface_detached(self) -> None:
        orbital_position_container = self.build_system("detached")
        assert testutils.surface_closed(
            faces=orbital_position_container.primary.faces,
            points=orbital_position_container.primary.points,
        )

    def test_closed_surface_semi_detached(self) -> None:
        orbital_position_container = self.build_system("semi-detached")
        assert testutils.surface_closed(
            faces=orbital_position_container.primary.faces,
            points=orbital_position_container.primary.points,
        )

    def test_closed_surface_over_contact(self) -> None:
        params = testutils.BINARY_SYSTEM_PARAMS["over-contact"].copy()
        params.update({"primary_discretization_factor": 9})
        s = testutils.prepare_binary_system(params)

        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)

        points = np.concatenate(
            (orbital_position_container.primary.points, orbital_position_container.secondary.points),
        )
        faces = np.concatenate(
            (
                orbital_position_container.primary.faces,
                orbital_position_container.secondary.faces + np.shape(orbital_position_container.primary.points)[0],
            ),
        )
        assert testutils.surface_closed(faces=faces, points=points)


class BuildSpottyFacesTestCase(ElisaTestCase):
    @staticmethod
    def build_system(key: str) -> OrbitalPositionContainer:
        params = testutils.BINARY_SYSTEM_PARAMS[key].copy()
        params.update({"primary_discretization_factor": 10})
        s = testutils.prepare_binary_system(
            params,
            spots_primary=testutils.SPOTS_META["primary"],
            spots_secondary=testutils.SPOTS_META["secondary"],
        )

        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        for component in ["primary", "secondary"]:
            instance = getattr(orbital_position_container, component)
            points = instance.points
            faces = instance.faces
            if isinstance(instance.spots, (dict,)):
                for spot in instance.spots.values():
                    faces = up.concatenate((faces, spot.faces + len(points)), axis=0)
                    points = up.concatenate((points, spot.points), axis=0)
            instance.points = points
            instance.faces = faces
        return orbital_position_container

    @staticmethod
    def generator_test_faces(key: str, length: list[int]) -> None:
        params = testutils.BINARY_SYSTEM_PARAMS[key].copy()
        params.update({"primary_discretization_factor": 10})
        s = testutils.prepare_binary_system(
            params,
            spots_primary=testutils.SPOTS_META["primary"],
            spots_secondary=testutils.SPOTS_META["secondary"],
        )

        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)

        assert_array_equal(
            [
                len(orbital_position_container.primary.faces),
                len(orbital_position_container.secondary.faces),
                len(orbital_position_container.primary.spots[0].faces),
                len(orbital_position_container.secondary.spots[0].faces),
            ],
            length,
        )

    def test_build_faces_detached(self) -> None:
        self.generator_test_faces("detached", [773, 516, 97, 24])

    def test_build_faces_over_contact(self) -> None:
        self.generator_test_faces("over-contact", [685, 434, 97, 24])

    def test_build_faces_over_contact2(self) -> None:
        settings.configure(MESH_GENERATOR="improved_trapezoidal")
        self.generator_test_faces("over-contact", [689, 440, 97, 24])

    def test_build_faces_semi_detached(self) -> None:
        self.generator_test_faces("semi-detached", [773, 516, 97, 24])

    def test_build_faces_semi_detached2(self) -> None:
        settings.configure(MESH_GENERATOR="improved_trapezoidal")
        self.generator_test_faces("semi-detached", [827, 550, 97, 24])

    def test_closed_surface_detached(self) -> None:
        orbital_position_container = self.build_system("detached")
        assert testutils.surface_closed(
            faces=orbital_position_container.primary.faces,
            points=orbital_position_container.primary.points,
        )

    def test_closed_surface_semi_detached(self) -> None:
        orbital_position_container = self.build_system("semi-detached")
        assert testutils.surface_closed(
            faces=orbital_position_container.primary.faces,
            points=orbital_position_container.primary.points,
        )

    def test_closed_surface_over_contact(self) -> None:
        params = testutils.BINARY_SYSTEM_PARAMS["over-contact"].copy()
        params.update({"primary_discretization_factor": 7})
        s = testutils.prepare_binary_system(
            params,
            spots_primary=testutils.SPOTS_META["primary"],
            spots_secondary=testutils.SPOTS_META["secondary"],
        )
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        for component in ["primary", "secondary"]:
            instance = getattr(orbital_position_container, component)
            points = instance.points
            faces = instance.faces
            if isinstance(instance.spots, (dict,)):
                for spot in instance.spots.values():
                    faces = up.concatenate((faces, spot.faces + len(points)), axis=0)
                    points = up.concatenate((points, spot.points), axis=0)
            instance.points = points
            instance.faces = faces

        points = np.concatenate(
            (orbital_position_container.primary.points, orbital_position_container.secondary.points),
        )
        faces = np.concatenate(
            (
                orbital_position_container.primary.faces,
                orbital_position_container.secondary.faces + np.shape(orbital_position_container.primary.points)[0],
            ),
        )
        assert testutils.surface_closed(faces=faces, points=points)


class BuildSurfaceAreasTestCase(ElisaTestCase):
    def generator_test_surface_areas(self, key: str, kind: str, less: float | None = None) -> None:
        params = testutils.BINARY_SYSTEM_PARAMS[key].copy()
        params.update({"primary_discretization_factor": 10})
        s = testutils.prepare_binary_system(
            params,
            spots_primary=testutils.SPOTS_META["primary"],
            spots_secondary=testutils.SPOTS_META["secondary"],
        )
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        orbital_position_container.build_surface_areas()

        if kind == "contain":
            assert not is_empty(orbital_position_container.primary.areas)
            assert not is_empty(orbital_position_container.secondary.areas)

            assert not is_empty(orbital_position_container.primary.spots[0].areas)
            assert not is_empty(orbital_position_container.secondary.spots[0].areas)

        if kind == "size":
            assert np.all(up.less(orbital_position_container.primary.areas, less))
            assert np.all(up.less(orbital_position_container.secondary.areas, less))

            assert np.all(up.less(orbital_position_container.primary.spots[0].areas, less))
            assert np.all(up.less(orbital_position_container.secondary.spots[0].areas, less))

    def test_build_surface_areas_detached(self) -> None:
        self.generator_test_surface_areas("detached", kind="contain")

    def test_build_surface_areas_over_contact(self) -> None:
        self.generator_test_surface_areas("over-contact", kind="contain")

    def test_build_surface_areas_semi_detached(self) -> None:
        self.generator_test_surface_areas("semi-detached", kind="contain")

    def test_build_surface_areas_detached_size(self) -> None:
        self.generator_test_surface_areas("detached", kind="size", less=5e-6)

    def test_build_surface_areas_over_contact_size(self) -> None:
        self.generator_test_surface_areas("over-contact", kind="size", less=8e-3)

    def test_build_surface_areas_semi_detached_size(self) -> None:
        self.generator_test_surface_areas("semi-detached", kind="size", less=8e-3)


class BuildSpottyFacesOrientationTestCase(ElisaTestCase):
    def generator_test_face_orientaion(self, key: str, kind: str) -> None:
        params = testutils.BINARY_SYSTEM_PARAMS[key].copy()
        params.update({"primary_discretization_factor": 7})
        s = testutils.prepare_binary_system(
            params,
            spots_primary=testutils.SPOTS_META["primary"],
            spots_secondary=testutils.SPOTS_META["secondary"],
        )
        orbital_position_container: OrbitalPositionContainer = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        orbital_position_container.build_surface_areas()
        orbital_position_container.build_faces_orientation(components_distance=1.0)

        if kind == "present":
            assert not is_empty(orbital_position_container.primary.normals)
            assert not is_empty(orbital_position_container.secondary.normals)

            assert not is_empty(orbital_position_container.primary.spots[0].normals)
            assert not is_empty(orbital_position_container.secondary.spots[0].normals)

        _assert = self.assertTrue
        if kind == "direction":
            o = orbital_position_container

            for component in ["primary", "secondary"]:
                star = getattr(o, component)

                face_points = star.points[star.faces]
                spot_face_points = star.spots[0].points[star.spots[0].faces]

                face_points[:, :, 0] = face_points[:, :, 0] - 1 if component == "secondary" else face_points[:, :, 0]
                spot_face_points[:, :, 0] = (
                    spot_face_points[:, :, 0] - 1 if component == "secondary" else spot_face_points[:, :, 0]
                )

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

        if kind == "size":
            o = orbital_position_container

            for component in ["primary", "secondary"]:
                star = getattr(o, component)

                normals_size = np.linalg.norm(star.normals, axis=1)
                _assert((np.round(normals_size, 5) == 1).all())

                spot_normals_size = np.linalg.norm(star.spots[0].normals, axis=1)
                _assert((np.round(spot_normals_size, 5) == 1).all())

    def test_if_normals_present_detached(self) -> None:
        self.generator_test_face_orientaion("detached", "present")

    def test_if_normals_present_semi_detached(self) -> None:
        self.generator_test_face_orientaion("semi-detached", "present")

    def test_if_normals_present_overcontact(self) -> None:
        self.generator_test_face_orientaion("over-contact", "present")

    def test_normals_direction_detached(self) -> None:
        self.generator_test_face_orientaion("detached", "direction")

    def test_normals_direction_semi_detached(self) -> None:
        self.generator_test_face_orientaion("semi-detached", "direction")

    def test_normals_direction_overcontact(self) -> None:
        self.generator_test_face_orientaion("over-contact", "direction")

    def test_normals_size_detached(self) -> None:
        self.generator_test_face_orientaion("detached", "size")

    def test_normals_size_semi_detached(self) -> None:
        self.generator_test_face_orientaion("semi-detached", "size")

    def test_normals_size_overcontact(self) -> None:
        self.generator_test_face_orientaion("over-contact", "size")
