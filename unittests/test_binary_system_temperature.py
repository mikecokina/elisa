# keep it first
# due to stupid astropy units/constants implementation
from unittests import set_astropy_units

import numpy as np
import os.path as op
from copy import copy

from elisa import umpy as up
from elisa import settings, BinarySystem
from elisa.utils import is_empty
from unittests import utils as testutils
from unittests.utils import ElisaTestCase

set_astropy_units()


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
        self.assertTrue("Limb darkening interpolation lead to numpy.nan/None value." in str(context.exception))

    def test_build_temperatures_detached_physical(self):
        self.generator_test_temperatures('detached-physical', [[4998, 5002], [4999, 5004]])

    def test_build_temperatures_over_contact(self):
        self.generator_test_temperatures('over-contact', [[4155, 5406], [4240, 5436]])

    def test_build_temperatures_semi_detached(self):
        self.generator_test_temperatures('semi-detached', [[3760, 5335], [3840, 5450]])


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


class GravityDarkeningAlbedoTestCase(ElisaTestCase):
    def setUp(self):
        super(GravityDarkeningAlbedoTestCase, self).setUp()
        settings.configure(**{
            "LIMB_DARKENING_LAW": "linear",
            "LD_TABLES": op.join(op.dirname(op.abspath(__file__)), "data", "light_curves", "limbdarkening")
        })
        self._base_model = {
              "system": {
                "inclination": 90.0,
                "period": 10.1,
                "argument_of_periastron": "90.0 deg",  # string representation of astropy quntity is also valid
                "gamma": 0.0,
                "eccentricity": 0.3,
                "primary_minimum_time": 0.0,
                "phase_shift": 0.0
              },
              "primary": {
                "mass": 2.0,
                "surface_potential": 100,
                "synchronicity": 1.0,
                "t_eff": 6500.0,
                "metallicity": 0.0
              },
              "secondary": {
                "mass": 1.0,
                "surface_potential": 100,
                "synchronicity": 1.0,
                "t_eff": 7500.0,
                "metallicity": 0.0
              }
            }

    def generator_test_betas(self, params, expected):
        s = BinarySystem.from_json(params)
        self.assertAlmostEqual(s.primary.gravity_darkening, expected[0], 5)
        self.assertAlmostEqual(s.secondary.gravity_darkening, expected[1], 5)

    def generator_test_albedo(self, params, expected):
        s = BinarySystem.from_json(params)
        self.assertAlmostEqual(s.primary.albedo, expected[0], 5)
        self.assertAlmostEqual(s.secondary.albedo, expected[1], 5)

    def test_user_defined(self):
        params = copy(self._base_model)
        params['primary'].update({'gravity_darkening': 1.0})
        params['secondary'].update({'gravity_darkening': 1.0})
        params['primary'].update({'albedo': 1.0})
        params['secondary'].update({'albedo': 1.0})

        self.generator_test_betas(params, (1.0, 1.0))
        self.generator_test_albedo(params, (1.0, 1.0))

    def test_auto_primary(self):
        params = copy(self._base_model)
        params['secondary'].update({'gravity_darkening': 1.0})
        params['primary'].update({'albedo': 1.0})

        self.generator_test_betas(params, (0.5302088, 1.0))
        self.generator_test_albedo(params, (1.0, 0.901515))

    def test_auto_both(self):
        self.generator_test_betas(self._base_model, (0.5302088, 0.968017))
        self.generator_test_albedo(self._base_model, (0.7126008, 0.901515))
