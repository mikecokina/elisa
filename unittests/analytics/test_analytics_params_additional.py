"""Additional comprehensive tests for analytics parameters module."""


import numpy as np

from elisa import units as u
from elisa.analytics.params import parameters
from elisa.base.error import InitialParamsError
from unittests import set_astropy_units
from unittests.utils import ElisaTestCase

set_astropy_units()


class InitialParameterTestCase(ElisaTestCase):
    """Unit tests for InitialParameter class."""

    def test_initialization_with_value(self):
        """Test basic initialization of InitialParameter with value."""
        from elisa.analytics.params.transform import StarInitialProperties

        param = parameters.InitialParameter(
            transform_cls=StarInitialProperties,
            param="t_eff",
            value=5000.0,
            unit=u.K,
            fixed=False,
            min=4000.0,
            max=6000.0,
        )

        self.assertEqual(param.param, "t_eff")
        self.assertEqual(param.value, 5000.0)
        self.assertFalse(param.fixed)
        self.assertIsNone(param.constraint)

    def test_initialization_with_constraint(self):
        """Test initialization with constraint instead of value."""
        from elisa.analytics.params.transform import BinaryInitialProperties

        param = parameters.InitialParameter(
            transform_cls=BinaryInitialProperties,
            param="semi_major_axis",
            value=None,
            constraint="16.0 / sin(system@inclination)",
            fixed=None,
            unit=u.solRad,
        )

        self.assertEqual(param.param, "semi_major_axis")
        self.assertIsNone(param.value)
        self.assertEqual(param.constraint, "16.0 / sin(system@inclination)")

    def test_unit_conversion(self):
        """Test that units are converted correctly."""
        from elisa.analytics.params.transform import BinaryInitialProperties

        param = parameters.InitialParameter(
            transform_cls=BinaryInitialProperties,
            param="gamma",
            value=10.0,
            unit=u.km / u.s,
            fixed=False,
            min=5.0,
            max=20.0,
        )

        # gamma should be converted from km/s to m/s (10 km/s = 10000 m/s)
        self.assertAlmostEqual(param.value, 10000.0, places=2)

    def test_fixed_parameter_clears_bounds(self):
        """Test that fixed parameters have None bounds and sigma."""
        from elisa.analytics.params.transform import StarInitialProperties

        param = parameters.InitialParameter(
            transform_cls=StarInitialProperties,
            param="gravity_darkening",
            value=0.32,
            unit=None,
            fixed=True,
            min=0.0,
            max=1.0,
            sigma=0.01,
        )

        self.assertIsNone(param.min)
        self.assertIsNone(param.max)
        self.assertIsNone(param.sigma)

    def test_copy_method(self):
        """Test that copy creates independent deep copy."""
        from elisa.analytics.params.transform import StarInitialProperties

        original = parameters.InitialParameter(
            transform_cls=StarInitialProperties,
            param="t_eff",
            value=5000.0,
            unit=u.K,
            fixed=False,
            min=4000.0,
            max=6000.0,
        )

        copied = original.copy()

        # Modify copy
        copied.value = 6000.0

        # Original should be unchanged
        self.assertEqual(original.value, 5000.0)
        self.assertEqual(copied.value, 6000.0)

    def test_to_dict_method(self):
        """Test conversion to dictionary."""
        from elisa.analytics.params.transform import StarInitialProperties

        param = parameters.InitialParameter(
            transform_cls=StarInitialProperties,
            param="t_eff",
            value=5000.0,
            unit=u.K,
            fixed=False,
            min=4000.0,
            max=6000.0,
            sigma=100.0,
        )

        param_dict = param.to_dict()

        self.assertEqual(param_dict["param"], "t_eff")
        self.assertEqual(param_dict["value"], 5000.0)
        self.assertFalse(param_dict["fixed"])
        self.assertEqual(param_dict["sigma"], 100.0)

    def test_repr_and_str(self):
        """Test string representation."""
        from elisa.analytics.params.transform import StarInitialProperties

        param = parameters.InitialParameter(
            transform_cls=StarInitialProperties,
            param="t_eff",
            value=5000.0,
            unit=u.K,
            fixed=False,
            min=4000.0,
            max=6000.0,
        )

        repr_str = repr(param)
        self.assertIn("t_eff", repr_str)
        self.assertIn("5000", repr_str)


class ParameterMetaTestCase(ElisaTestCase):
    """Unit tests for ParameterMeta class."""

    def test_initialization(self):
        """Test ParameterMeta initialization."""
        meta = parameters.ParameterMeta(
            param="t_eff",
            value=5000.0,
            unit=u.K,
            fixed=False,
            min=4000.0,
            max=6000.0,
            sigma=100.0,
            constraint=None,
        )

        self.assertEqual(meta.param, "t_eff")
        self.assertEqual(meta.value, 5000.0)
        self.assertEqual(meta.unit, u.K)
        self.assertFalse(meta.fixed)

    def test_to_dict_method(self):
        """Test ParameterMeta to_dict conversion."""
        meta = parameters.ParameterMeta(
            param="t_eff",
            value=5000.0,
            unit=u.K,
            fixed=False,
            min=4000.0,
            max=6000.0,
            sigma=100.0,
            constraint=None,
        )

        meta_dict = meta.to_dict()

        self.assertEqual(meta_dict["param"], "t_eff")
        self.assertEqual(meta_dict["value"], 5000.0)
        self.assertFalse(meta_dict["fixed"])


class SpotInitialParametersTestCase(ElisaTestCase):
    """Unit tests for SpotInitialParameters class."""

    def test_initialization(self):
        """Test SpotInitialParameters initialization."""
        spot_params = {
            "label": "spot1",
            "latitude": {"value": 30.0, "fixed": False, "min": 0, "max": 90},
            "longitude": {"value": 45.0, "fixed": False, "min": 0, "max": 360},
            "angular_radius": {"value": 15.0, "fixed": True},
            "temperature_factor": {"value": 0.95, "fixed": True},
        }

        spot = parameters.SpotInitialParameters(**spot_params)

        self.assertIn("spot", spot.label)
        self.assertIn("spot1", spot.label)
        self.assertIsInstance(spot.latitude, parameters.InitialParameter)
        self.assertEqual(spot.latitude.value, 30.0)

    def test_slot_attributes(self):
        """Test that all slots are properly initialized."""
        spot_params = {
            "label": "spot1",
            "latitude": {"value": 30.0, "fixed": False},
            "longitude": {"value": 45.0, "fixed": False},
            "angular_radius": {"value": 15.0, "fixed": True},
            "temperature_factor": {"value": 0.95, "fixed": True},
        }

        spot = parameters.SpotInitialParameters(**spot_params)

        expected_slots = ["longitude", "latitude", "angular_radius", "temperature_factor", "label"]
        for slot in expected_slots:
            self.assertTrue(hasattr(spot, slot))

    def test_validity_check_bounds(self):
        """Test validity check for parameter bounds."""
        invalid_spot_params = {
            "label": "spot1",
            "latitude": {"value": 100.0, "fixed": False, "min": 0, "max": 90},  # value > max
            "longitude": {"value": 45.0, "fixed": False, "min": 0, "max": 360},
            "angular_radius": {"value": 15.0, "fixed": True},
            "temperature_factor": {"value": 0.95, "fixed": True},
        }

        with self.assertRaises(InitialParamsError):
            parameters.SpotInitialParameters(**invalid_spot_params)


class PulsationInitialParametersTestCase(ElisaTestCase):
    """Unit tests for PulsationInitialParameters class."""

    def test_initialization(self):
        """Test PulsationInitialParameters initialization."""
        pulsation_params = {
            "label": "mode1",
            "l": {"value": 2, "fixed": True},
            "m": {"value": 1, "fixed": False},
            "amplitude": {"value": 1.0, "fixed": False, "min": 0.0, "max": 5.0},
            "frequency": {"value": 10.0, "fixed": False, "min": 1.0, "max": 20.0},
            "start_phase": {"value": 0.0, "fixed": True},
            "mode_axis_theta": {"value": 0.0, "fixed": True},
            "mode_axis_phi": {"value": 0.0, "fixed": True},
        }

        pulsation = parameters.PulsationInitialParameters(**pulsation_params)

        self.assertIn("pulsation", pulsation.label)
        self.assertIn("mode1", pulsation.label)
        self.assertEqual(pulsation.l.value, 2)
        self.assertEqual(pulsation.m.value, 1)

    def test_default_mode_axis_values(self):
        """Test that mode_axis angles default to 0."""
        pulsation_params = {
            "label": "mode1",
            "l": {"value": 2, "fixed": True},
            "m": {"value": 1, "fixed": False},
            "amplitude": {"value": 1.0, "fixed": False},
            "frequency": {"value": 10.0, "fixed": False},
        }

        pulsation = parameters.PulsationInitialParameters(**pulsation_params)

        self.assertEqual(pulsation.mode_axis_phi, 0.0)
        self.assertEqual(pulsation.mode_axis_theta, 0.0)

    def test_slot_attributes(self):
        """Test that all slots are properly initialized."""
        pulsation_params = {
            "label": "mode1",
            "l": {"value": 2, "fixed": True},
            "m": {"value": 1, "fixed": False},
            "amplitude": {"value": 1.0, "fixed": False},
            "frequency": {"value": 10.0, "fixed": False},
            "start_phase": {"value": 0.0, "fixed": True},
            "mode_axis_theta": {"value": 0.0, "fixed": True},
            "mode_axis_phi": {"value": 0.0, "fixed": True},
        }

        pulsation = parameters.PulsationInitialParameters(**pulsation_params)

        expected_slots = ["l", "m", "amplitude", "frequency", "start_phase", "mode_axis_theta", "mode_axis_phi", "label"]
        for slot in expected_slots:
            self.assertTrue(hasattr(pulsation, slot))


class StarInitialParametersTestCase(ElisaTestCase):
    """Unit tests for StarInitialParameters class."""

    def test_initialization_without_phenomena(self):
        """Test StarInitialParameters initialization without spots or pulsations."""
        star_params = {
            "t_eff": {"value": 5000.0, "fixed": False, "min": 4000.0, "max": 6000.0},
            "surface_potential": {"value": 3.0, "fixed": False, "min": 2.5, "max": 3.5},
            "gravity_darkening": {"value": 0.32, "fixed": True},
            "albedo": {"value": 0.6, "fixed": True},
        }

        star = parameters.StarInitialParameters(**star_params)

        self.assertEqual(star.t_eff.value, 5000.0)
        self.assertFalse(hasattr(star, "spots") and star.spots)
        self.assertFalse(hasattr(star, "pulsations") and star.pulsations)

    def test_initialization_with_spots(self):
        """Test StarInitialParameters initialization with spots."""
        star_params = {
            "t_eff": {"value": 5000.0, "fixed": False},
            "surface_potential": {"value": 3.0, "fixed": False},
            "gravity_darkening": {"value": 0.32, "fixed": True},
            "albedo": {"value": 0.6, "fixed": True},
            "spots": [
                {
                    "label": "spot1",
                    "latitude": {"value": 30.0, "fixed": False},
                    "longitude": {"value": 45.0, "fixed": False},
                    "angular_radius": {"value": 15.0, "fixed": True},
                    "temperature_factor": {"value": 0.95, "fixed": True},
                },
            ],
        }

        star = parameters.StarInitialParameters(**star_params)

        self.assertTrue(hasattr(star, "spots") and star.spots)
        self.assertEqual(len(star.spots), 1)
        self.assertIsInstance(star.spots[0], parameters.SpotInitialParameters)

    def test_initialization_with_pulsations(self):
        """Test StarInitialParameters initialization with pulsations."""
        star_params = {
            "t_eff": {"value": 5000.0, "fixed": False},
            "surface_potential": {"value": 3.0, "fixed": False},
            "gravity_darkening": {"value": 0.32, "fixed": True},
            "albedo": {"value": 0.6, "fixed": True},
            "pulsations": [
                {
                    "label": "mode1",
                    "l": {"value": 2, "fixed": True},
                    "m": {"value": 1, "fixed": False},
                    "amplitude": {"value": 1.0, "fixed": False},
                    "frequency": {"value": 10.0, "fixed": False},
                },
            ],
        }

        star = parameters.StarInitialParameters(**star_params)

        self.assertTrue(hasattr(star, "pulsations") and star.pulsations)
        self.assertEqual(len(star.pulsations), 1)
        self.assertIsInstance(star.pulsations[0], parameters.PulsationInitialParameters)


class NuisanceInitialParametersTestCase(ElisaTestCase):
    """Unit tests for NuisanceInitialParameters class."""

    def test_initialization(self):
        """Test NuisanceInitialParameters initialization."""
        nuisance_params = {
            "ln_f": {"value": -20.0, "fixed": True},
        }

        nuisance = parameters.NuisanceInitialPrameters(**nuisance_params)

        self.assertEqual(nuisance.ln_f.value, -20.0)
        self.assertTrue(nuisance.ln_f.fixed)

    def test_slot_attributes(self):
        """Test that all slots are properly initialized."""
        nuisance_params = {
            "ln_f": {"value": -20.0, "fixed": True},
        }

        nuisance = parameters.NuisanceInitialPrameters(**nuisance_params)

        self.assertTrue(hasattr(nuisance, "ln_f"))


class UtilityFunctionsTestCase(ElisaTestCase):
    """Unit tests for utility functions."""

    def test_renormalize_value(self):
        """Test renormalize_value function."""
        result = parameters.renormalize_value(0.5, 5000.0, 7000.0)
        expected = 6000.0
        self.assertAlmostEqual(result, expected)

    def test_normalize_value(self):
        """Test normalize_value function."""
        result = parameters.normalize_value(6000.0, 5000.0, 7000.0)
        expected = 0.5
        self.assertAlmostEqual(result, expected)

    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize->renormalize gives original value."""
        original = 6000.0
        min_val, max_val = 5000.0, 7000.0

        normalized = parameters.normalize_value(original, min_val, max_val)
        denormalized = parameters.renormalize_value(normalized, min_val, max_val)

        self.assertAlmostEqual(original, denormalized, places=10)

    def test_extend_json_with_atm_params_atmosphere(self):
        """Test extending JSON with atmosphere parameters."""
        params = {
            "primary@t_eff": {"value": 5000.0},
            "secondary@t_eff": {"value": 4000.0},
        }

        atmosphere_model = {
            "primary": "bb",
            "secondary": "ck04",
        }

        extended = parameters.extend_json_with_atm_params(params, atmosphere_model=atmosphere_model)

        self.assertEqual(extended["primary@atmosphere"], "bb")
        self.assertEqual(extended["secondary@atmosphere"], "ck04")

    def test_extend_json_with_atm_params_limb_darkening(self):
        """Test extending JSON with limb darkening coefficients."""
        params = {
            "primary@t_eff": {"value": 5000.0},
            "secondary@t_eff": {"value": 4000.0},
        }

        limb_darkening = {
            "primary": {"bolometric": [0.5, 0.5]},
            "secondary": {"bolometric": [0.6, 0.4]},
        }

        extended = parameters.extend_json_with_atm_params(
            params,
            limb_darkening_coefficients=limb_darkening,
        )

        self.assertEqual(extended["primary@limb_darkening_coefficients"], {"bolometric": [0.5, 0.5]})
        self.assertEqual(extended["secondary@limb_darkening_coefficients"], {"bolometric": [0.6, 0.4]})

    def test_extend_json_with_invalid_component(self):
        """Test that extending with invalid component raises error."""
        params = {
            "primary@t_eff": {"value": 5000.0},
        }

        atmosphere_model = {
            "tertiary": "bb",  # invalid component
        }

        with self.assertRaises(ValueError):
            parameters.extend_json_with_atm_params(params, atmosphere_model=atmosphere_model)

    def test_prepare_nuisance_properties_set(self):
        """Test prepare_nuisance_properties_set function."""
        xn = np.array([-20.0])
        properties = ["nuisance@ln_f"]
        fixed = {"nuisance@ln_f": 0.0}

        result = parameters.prepare_nuisance_properties_set(xn, properties, fixed)

        self.assertIn("nuisance@ln_f", result)
        self.assertEqual(result["nuisance@ln_f"], 0.0)

    def test_check_for_invalid_constraint(self):
        """Test check_for_invalid_constraint function - checks after substitution."""
        # After substitution with values, remaining @ symbols indicate invalid params
        constrained = {
            "system@semi_major_axis": "16.0 / sin(90.0)",  # already substituted, no @ symbols
        }
        allowed_params = ["system@inclination", "system@period"]

        # Should not raise since no @ symbols remain
        parameters.check_for_invalid_constraint(constrained, allowed_params)

    def test_check_for_invalid_constraint_invalid_param(self):
        """Test check_for_invalid_constraint with invalid parameter."""
        constrained = {
            "system@semi_major_axis": "16.0 / sin(invalid@parameter)",
        }
        allowed_params = ["system@inclination", "system@period"]

        with self.assertRaises(InitialParamsError):
            parameters.check_for_invalid_constraint(constrained, allowed_params)

    def test_constraints_evaluator_simple(self):
        """Test constraints_evaluator with simple constraint."""
        substitution = {
            "primary@surface_potential": 3.0,
        }
        constrained = {
            "secondary@surface_potential": "2.0 * primary@surface_potential",
        }

        evaluated = parameters.constraints_evaluator(substitution, constrained)

        self.assertEqual(evaluated["secondary@surface_potential"], 6.0)

    def test_constraints_evaluator_with_numpy_function(self):
        """Test constraints_evaluator with numpy functions."""
        substitution = {
            "system@inclination": 90.0,
        }
        constrained = {
            "system@semi_major_axis": "16.0 / sin(radians(system@inclination))",
        }

        evaluated = parameters.constraints_evaluator(substitution, constrained)

        # sin(90 degrees) = 1, so 16.0 / 1 = 16.0
        self.assertAlmostEqual(evaluated["system@semi_major_axis"], 16.0, places=5)

    def test_extend_result_with_sma_community_format(self):
        """Test extend_result_with_sma for community format."""
        fit_params = {
            "system": {
                "mass_ratio": {"value": 0.5},
                "period": {"value": 3.0},
                "inclination": {"value": 85.0},
            },
            "primary": {"surface_potential": {"value": 3.0}},
            "secondary": {"surface_potential": {"value": 3.0}},
        }

        extended = parameters.extend_result_with_sma(fit_params)

        self.assertIn("semi_major_axis", extended["system"])
        self.assertTrue(extended["system"]["semi_major_axis"]["fixed"])

    def test_extend_result_with_sma_already_present(self):
        """Test extend_result_with_sma when sma is already present."""
        fit_params = {
            "system": {
                "semi_major_axis": {"value": 16.5},
                "mass_ratio": {"value": 0.5},
            },
        }

        extended = parameters.extend_result_with_sma(fit_params)

        # Should return unchanged
        self.assertEqual(extended["system"]["semi_major_axis"]["value"], 16.5)

    def test_extend_result_with_sma_standard_format(self):
        """Test extend_result_with_sma for standard format without mass_ratio."""
        fit_params = {
            "system": {
                "period": {"value": 3.0},
            },
        }

        extended = parameters.extend_result_with_sma(fit_params)

        # Should return unchanged
        self.assertNotIn("semi_major_axis", extended["system"])

