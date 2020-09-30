import numpy as np
import os.path as op

from elisa import umpy as up
from elisa import settings
from elisa.utils import is_empty
from unittests import utils as testutils
from unittests.utils import ElisaTestCase


class BuildSpotFreeTemperatureTestCase(ElisaTestCase):
    def generator_test_temperatures(self, key, allowed_range=None):

        settings.configure(**{
            "LIMB_DARKENING_LAW": "linear",
            "LD_TABLES": op.join(op.dirname(op.abspath(__file__)), "data", "light_curves", "limbdarkening")
        })

        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key])
        s.primary.discretization_factor = up.radians(10)
        s.secondary.discretization_factor = up.radians(10)
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        orbital_position_container.build_surface_areas()
        orbital_position_container.build_surface_gravity(components_distance=1.0)
        orbital_position_container.build_faces_orientation(components_distance=1.0)
        orbital_position_container.build_temperature_distribution(components_distance=1.0)

        print()
        i = np.argsort(orbital_position_container.secondary.temperatures)
        print(orbital_position_container.secondary.points[orbital_position_container.secondary.faces[i[:10]]])

        print()
        if allowed_range:
            obtained_primary = [np.min(orbital_position_container.primary.temperatures),
                                np.max(orbital_position_container.primary.temperatures)]
            obtained_secondary = [np.min(orbital_position_container.secondary.temperatures),
                                  np.max(orbital_position_container.secondary.temperatures)]

            self.assertTrue((obtained_primary[0] >= allowed_range[0][0]) &
                            (obtained_primary[1] <= allowed_range[0][1]))
            self.assertTrue((obtained_secondary[0] >= allowed_range[1][0]) &
                            (obtained_secondary[1] <= allowed_range[1][1]))

    def test_build_temperatures_detached(self):
        with self.assertRaises(Exception) as context:
            self.generator_test_temperatures('detached')
        self.assertTrue("It might be caused by definition of unphysical object on input" in str(context.exception))

    def test_build_temperatures_detached_physical(self):
        self.generator_test_temperatures('detached-physical', [[4998, 5002], [4999, 5004]])

    def test_build_temperatures_over_contact(self):
        self.generator_test_temperatures('over-contact', [[4155, 5405], [4240, 5435]])

    def test_build_temperatures_semi_detached(self):
        self.generator_test_temperatures('semi-detached', [[3760, 5335], [3865, 5450]])


class BuildSpottyTemperatureTestCase(ElisaTestCase):
    def setUp(self):
        super(BuildSpottyTemperatureTestCase, self).setUp()
        settings.configure(**{
            "LIMB_DARKENING_LAW": "linear",
            "LD_TABLES": op.join(op.dirname(op.abspath(__file__)), "data", "light_curves", "limbdarkening")
        })
    
    def generator_test_temperatures(self, key):
        s = testutils.prepare_binary_system(testutils.BINARY_SYSTEM_PARAMS[key],
                                            spots_primary=testutils.SPOTS_META["primary"],
                                            spots_secondary=testutils.SPOTS_META["secondary"]
                                            )
        s.primary.discretization_factor = up.radians(10)
        s.secondary.discretization_factor = up.radians(10)
        orbital_position_container = testutils.prepare_orbital_position_container(s)
        orbital_position_container.build_mesh(components_distance=1.0)
        orbital_position_container.build_faces(components_distance=1.0)
        orbital_position_container.build_surface_areas()
        orbital_position_container.build_surface_gravity(components_distance=1.0)
        orbital_position_container.build_faces_orientation(components_distance=1.0)
        orbital_position_container.build_temperature_distribution(components_distance=1.0)

        self.assertTrue(not is_empty(orbital_position_container.primary.spots[0].temperatures))
        self.assertTrue(not is_empty(orbital_position_container.secondary.spots[0].temperatures))

    def test_build_temperatures_detached_physical(self):
        self.generator_test_temperatures('detached-physical')

    def test_build_temperatures_over_contact(self):
        self.generator_test_temperatures('over-contact')

    def test_build_temperatures_semi_detached(self):
        self.generator_test_temperatures('semi-detached')
