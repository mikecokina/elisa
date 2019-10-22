import numpy as np
from numpy.testing import assert_array_equal

from elisa import umpy as up
from elisa.base.container import StarContainer
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.const import BINARY_POSITION_PLACEHOLDER
from elisa.utils import is_empty
from unittests import utils as testutils
from unittests.utils import ElisaTestCase


class BuildFacesSpotsFreeTestCase(ElisaTestCase):
    @staticmethod
    def generator_test_faces(key, d, length):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key])
        s.primary.discretization_factor = d
        s.secondary.discretization_factor = d

        orbital_position_container = OrbitalPositionContainer(
            primary=StarContainer.from_properties_container(s.primary.to_properties_container()),
            secondary=StarContainer.from_properties_container(s.secondary.to_properties_container()),
            position=BINARY_POSITION_PLACEHOLDER(*(0, 1.0, 0.0, 0.0, 0.0)),
            **s.properties_serializer()
        )
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(component_distance=1.0)

        assert_array_equal([len(orbital_position_container.primary.faces),
                            len(orbital_position_container.secondary.faces)], length)

    def test_build_faces_detached(self):
        self.generator_test_faces('detached', up.radians(10), [848, 848])

    def test_build_faces_over_contact(self):
        self.generator_test_faces('over-contact', up.radians(10), [812, 784])

    def test_build_faces_semi_detached(self):
        self.generator_test_faces('semi-detached', up.radians(10), [848, 848])


class BuildSpottyFacesTestCase(ElisaTestCase):
    @staticmethod
    def generator_test_faces(key, d, length):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = d
        s.secondary.discretization_factor = d
        orbital_position_container = OrbitalPositionContainer(
            primary=StarContainer.from_properties_container(s.primary.to_properties_container()),
            secondary=StarContainer.from_properties_container(s.secondary.to_properties_container()),
            position=BINARY_POSITION_PLACEHOLDER(*(0, 1.0, 0.0, 0.0, 0.0)),
            **s.properties_serializer()
        )
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(component_distance=1.0)

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


class BuildSurfaceAreasTestCase(ElisaTestCase):
    def generator_test_surface_areas(self, key, d, kind, less=None):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = d
        s.secondary.discretization_factor = d
        orbital_position_container = OrbitalPositionContainer(
            primary=StarContainer.from_properties_container(s.primary.to_properties_container()),
            secondary=StarContainer.from_properties_container(s.secondary.to_properties_container()),
            position=BINARY_POSITION_PLACEHOLDER(*(0, 1.0, 0.0, 0.0, 0.0)),
            **s.properties_serializer()
        )
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(component_distance=1.0)
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
