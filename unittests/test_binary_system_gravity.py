import numpy as np

from elisa.utils import is_empty
from unittests import utils as testutils
from unittests.utils import ElisaTestCase
from elisa import umpy as up


class BuildSpotlessGravityTestCase(ElisaTestCase):
    def generator_test_gravity(self, key, over):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key])
        s.primary.discretization_factor = up.radians(10)
        s.secondary.discretization_factor = up.radians(10)

        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(component_distance=1.0)
        orbital_position_container.build_surface_areas()
        orbital_position_container.build_faces_orientation(components_distance=1.0)
        orbital_position_container.build_surface_gravity(components_distance=1.0)

        self.assertTrue(np.all(orbital_position_container.primary.log_g > over[0]))
        self.assertTrue(np.all(orbital_position_container.secondary.log_g > over[1]))

    def test_build_gravity_detached(self):
        self.generator_test_gravity('detached', over=[5.1, 5.4])

    def test_build_gravity_semi_detached(self):
        self.generator_test_gravity('semi-detached', over=[1.2, 1.2])

    def test_build_gravity_overcontact(self):
        self.generator_test_gravity('over-contact', over=[1.4, 1.3])


class BuildSpotGravityTestCase(ElisaTestCase):
    def generator_test_gravity(self, key):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"])
        s.primary.discretization_factor = up.radians(10)
        s.secondary.discretization_factor = up.radians(10)
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(component_distance=1.0)
        orbital_position_container.build_surface_areas()
        orbital_position_container.build_faces_orientation(components_distance=1.0)
        orbital_position_container.build_surface_gravity(components_distance=1.0)

        self.assertTrue(hasattr(orbital_position_container.primary.spots[0], "potential_gradient_magnitudes"))
        self.assertTrue(hasattr(orbital_position_container.secondary.spots[0], "potential_gradient_magnitudes"))

        self.assertTrue(hasattr(orbital_position_container.primary.spots[0], "log_g"))
        self.assertTrue(hasattr(orbital_position_container.secondary.spots[0], "log_g"))

        self.assertTrue(not is_empty(orbital_position_container.primary.spots[0].log_g))
        self.assertTrue(not is_empty(orbital_position_container.secondary.spots[0].log_g))

    def test_build_gravity_detached(self):
        self.generator_test_gravity('detached')

    def test_build_gravity_semi_detached(self):
        self.generator_test_gravity('semi-detached')

    def test_build_gravity_overcontact(self):
        self.generator_test_gravity('over-contact')
