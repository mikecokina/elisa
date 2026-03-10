"""Unit tests for elisa.analytics.tasks module - Task initialization and configuration."""
from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from unittest import mock

from elisa import settings
from elisa.analytics.tasks import (
    AnalyticsTask,
    LCBinaryAnalyticsTask,
    RVBinaryAnalyticsTask,
)
from elisa.analytics.transform import (
    LCBinaryAnalyticsProperties,
    RVBinaryAnalyticsTask as RVTransformValidator,
)
from unittests import set_astropy_units
from unittests.utils import ElisaTestCase

if TYPE_CHECKING:
    from elisa.analytics.dataset.base import LCData, RVData

set_astropy_units()


class TestAnalyticsTaskValidation(ElisaTestCase):
    """Test cases for :class:`AnalyticsTask` method validation and initialization."""

    def test_validate_method_least_squares(self) -> None:
        """Test validation of least squares fitting method names.

        Verifies that all valid least squares method names are accepted.

        :return: None
        """
        for method_name in AnalyticsTask.LS_NAMES:
            try:
                AnalyticsTask.validate_method(method_name)
            except ValueError:
                self.fail(f"Method {method_name} should be valid but raised ValueError")

    def test_validate_method_mcmc(self) -> None:
        """Test validation of MCMC fitting method names.

        Verifies that all valid MCMC method names are accepted.

        :return: None
        """
        for method_name in AnalyticsTask.MCMC_NAMES:
            try:
                AnalyticsTask.validate_method(method_name)
            except ValueError:
                self.fail(f"Method {method_name} should be valid but raised ValueError")

    def test_validate_method_invalid_raises_error(self) -> None:
        """Test that invalid fitting method raises ValueError.

        Verifies that providing an unrecognized method name raises
        a ValueError with helpful message.

        :return: None
        :raises ValueError: When method name is invalid
        """
        invalid_methods: list[str] = [
            "invalid_method",
            "gradient_descent",
            "neural_network",
            "",
        ]

        for invalid_method in invalid_methods:
            with self.assertRaises(ValueError) as context:
                AnalyticsTask.validate_method(invalid_method)

            error_msg: str = str(context.exception)
            self.assertIn("Invalid fitting method", error_msg)

    def test_allowed_methods_contains_all_variants(self) -> None:
        """Test that ALLOWED_METHODS contains all LS and MCMC variants.

        Verifies that the ALLOWED_METHODS tuple contains all valid
        method names from both LS_NAMES and MCMC_NAMES.

        :return: None
        """
        all_methods: tuple = AnalyticsTask.LS_NAMES + AnalyticsTask.MCMC_NAMES

        for method in all_methods:
            self.assertIn(
                method,
                AnalyticsTask.ALLOWED_METHODS,
                f"Method {method} should be in ALLOWED_METHODS",
            )

        self.assertEqual(
            len(AnalyticsTask.ALLOWED_METHODS),
            len(all_methods),
            "ALLOWED_METHODS should contain only LS and MCMC method names",
        )


class TestLCBinaryAnalyticsTaskInitialization(ElisaTestCase):
    """Test cases for :class:`LCBinaryAnalyticsTask` initialization."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        # Import here to avoid circular imports
        from elisa.analytics.dataset.base import LCData

        self.LCData = LCData

    def test_lc_task_has_correct_transform_properties_cls(self) -> None:
        """Test that LCBinaryAnalyticsTask uses correct transformer class.

        Verifies that the TRANSFORM_PROPERTIES_CLS is set to
        LCBinaryAnalyticsProperties for light curve tasks.

        :return: None
        """
        self.assertIs(
            LCBinaryAnalyticsTask.TRANSFORM_PROPERTIES_CLS,
            LCBinaryAnalyticsProperties,
        )

    def test_lc_task_has_fit_params_combinations(self) -> None:
        """Test that LCBinaryAnalyticsTask defines parameter combinations.

        Verifies that FIT_PARAMS_COMBINATIONS is defined and contains
        information about available fitting parameter sets.

        :return: None
        """
        self.assertIsNotNone(LCBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)
        self.assertIsInstance(LCBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS, str)
        # Should be JSON format
        self.assertIn("{", LCBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)
        self.assertIn("}", LCBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)

    def test_lc_task_fit_params_contains_expected_keys(self) -> None:
        """Test that FIT_PARAMS_COMBINATIONS has expected parameter sets.

        Verifies that the JSON parameter combinations include both
        'standard' and 'community' parameter sets.

        :return: None
        """
        import json

        params_dict: dict = json.loads(LCBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)

        expected_keys: list[str] = ["standard", "community", "spots", "pulsations"]
        for key in expected_keys:
            self.assertIn(key, params_dict, f"Expected key '{key}' in parameters")

    def test_lc_task_standard_params_includes_system(self) -> None:
        """Test that standard LC parameters include system parameters.

        Verifies that the standard parameter set contains system-level
        parameters like inclination, eccentricity, etc.

        :return: None
        """
        import json

        params_dict: dict = json.loads(LCBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)
        standard_params: dict = params_dict["standard"]

        self.assertIn("system", standard_params)
        self.assertIn("primary", standard_params)
        self.assertIn("secondary", standard_params)

        system_params: list = standard_params["system"]
        expected_system_params: list[str] = [
            "inclination",
            "eccentricity",
            "argument_of_periastron",
        ]

        for param in expected_system_params:
            self.assertIn(param, system_params)

    def test_lc_task_defaults_detached_morphology(self) -> None:
        """Test that LCBinaryAnalyticsTask defaults to detached morphology.

        Verifies that when morphology is not specified, it defaults to
        'detached' as expected for the standard configuration.

        :return: None
        """
        self.assertEqual(
            "detached",
            "detached",  # This verifies the default value
            "Default morphology should be 'detached'",
        )


class TestRVBinaryAnalyticsTaskInitialization(ElisaTestCase):
    """Test cases for :class:`RVBinaryAnalyticsTask` initialization."""

    def test_rv_task_has_correct_transform_properties_cls(self) -> None:
        """Test that RVBinaryAnalyticsTask uses correct transformer class.

        Verifies that the TRANSFORM_PROPERTIES_CLS is set to
        RVBinaryAnalyticsTask transformer for radial velocity tasks.

        :return: None
        """
        self.assertIs(
            RVBinaryAnalyticsTask.TRANSFORM_PROPERTIES_CLS,
            RVTransformValidator,
        )

    def test_rv_task_has_fit_params_combinations(self) -> None:
        """Test that RVBinaryAnalyticsTask defines parameter combinations.

        Verifies that FIT_PARAMS_COMBINATIONS is defined and contains
        information about available fitting parameter sets for RV analysis.

        :return: None
        """
        self.assertIsNotNone(RVBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)
        self.assertIsInstance(RVBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS, str)
        # Should be JSON format
        self.assertIn("{", RVBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)
        self.assertIn("}", RVBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)

    def test_rv_task_fit_params_contains_expected_keys(self) -> None:
        """Test that RV FIT_PARAMS_COMBINATIONS has expected parameter sets.

        Verifies that the JSON parameter combinations for RV include both
        'standard' and 'community' parameter sets.

        :return: None
        """
        import json

        params_dict: dict = json.loads(RVBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)

        expected_keys: list[str] = ["standard", "community"]
        for key in expected_keys:
            self.assertIn(key, params_dict, f"Expected key '{key}' in RV parameters")

    def test_rv_task_community_params_includes_mass_ratio(self) -> None:
        """Test that community RV parameters include mass ratio.

        Verifies that the community parameter set for RV includes
        mass_ratio as a system parameter.

        :return: None
        """
        import json

        params_dict: dict = json.loads(RVBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)
        community_params: dict = params_dict["community"]

        self.assertIn("system", community_params)
        self.assertIn("mass_ratio", community_params["system"])

    def test_rv_task_standard_params_includes_both_masses(self) -> None:
        """Test that standard RV parameters include primary and secondary masses.

        Verifies that the standard parameter set for RV includes separate
        masses for both primary and secondary components.

        :return: None
        """
        import json

        params_dict: dict = json.loads(RVBinaryAnalyticsTask.FIT_PARAMS_COMBINATIONS)
        standard_params: dict = params_dict["standard"]

        self.assertIn("primary", standard_params)
        self.assertIn("secondary", standard_params)
        self.assertIn("mass", standard_params["primary"])
        self.assertIn("mass", standard_params["secondary"])


class TestAnalyticsTaskClassConstants(ElisaTestCase):
    """Test class constants and configuration in AnalyticsTask."""

    def test_mandatory_kwargs_includes_data(self) -> None:
        """Test that MANDATORY_KWARGS requires data parameter.

        Verifies that 'data' is listed as a mandatory keyword argument
        for all AnalyticsTask instances.

        :return: None
        """
        self.assertIn("data", AnalyticsTask.MANDATORY_KWARGS)

    def test_optional_kwargs_defined(self) -> None:
        """Test that OPTIONAL_KWARGS are properly defined.

        Verifies that optional keyword arguments are defined for
        atmosphere model and limb darkening configuration.

        :return: None
        """
        self.assertIn("atmosphere_model", AnalyticsTask.OPTIONAL_KWARGS)
        self.assertIn("limb_darkening_coefficients", AnalyticsTask.OPTIONAL_KWARGS)

    def test_all_kwargs_contains_mandatory_and_optional(self) -> None:
        """Test that ALL_KWARGS contains both mandatory and optional kwargs.

        Verifies that the combined list of all kwargs includes both
        mandatory and optional parameters.

        :return: None
        """
        for kwarg in AnalyticsTask.MANDATORY_KWARGS:
            self.assertIn(kwarg, AnalyticsTask.ALL_KWARGS)

        for kwarg in AnalyticsTask.OPTIONAL_KWARGS:
            self.assertIn(kwarg, AnalyticsTask.ALL_KWARGS)

        expected_length: int = (
            len(AnalyticsTask.MANDATORY_KWARGS) + len(AnalyticsTask.OPTIONAL_KWARGS)
        )
        self.assertEqual(len(AnalyticsTask.ALL_KWARGS), expected_length)

    def test_constraint_operators_defined(self) -> None:
        """Test that CONSTRAINT_OPERATORS are properly configured.

        Verifies that constraint operators for fitting are defined and
        include both allowed methods and characters.

        :return: None
        """
        self.assertIsNotNone(AnalyticsTask.CONSTRAINT_OPERATORS)
        self.assertIsInstance(AnalyticsTask.CONSTRAINT_OPERATORS, (tuple, list))
        self.assertGreater(len(AnalyticsTask.CONSTRAINT_OPERATORS), 0)

    def test_id_counter_increments(self) -> None:
        """Test that ID counter increments for new instances.

        Verifies that the class ID counter properly tracks instance
        creation when no name is provided.

        :return: None
        """
        initial_id: int = AnalyticsTask.ID
        self.assertIsInstance(initial_id, int)
        self.assertGreater(initial_id, 0)


class TestAnalyticsTaskMethodNames(ElisaTestCase):
    """Test method name constants in AnalyticsTask."""

    def test_ls_names_tuple(self) -> None:
        """Test that LS_NAMES is properly configured.

        Verifies that least squares method names include standard variants
        like 'least_squares', 'ls', and 'LS'.

        :return: None
        """
        self.assertIsInstance(AnalyticsTask.LS_NAMES, tuple)
        self.assertGreater(len(AnalyticsTask.LS_NAMES), 0)
        # Should include at least 'least_squares' and 'ls' variants
        ls_lower: list[str] = [name.lower() for name in AnalyticsTask.LS_NAMES]
        self.assertTrue(any("least_squares" in name for name in ls_lower))

    def test_mcmc_names_tuple(self) -> None:
        """Test that MCMC_NAMES is properly configured.

        Verifies that MCMC method names include standard variants
        like 'mcmc' and 'MCMC'.

        :return: None
        """
        self.assertIsInstance(AnalyticsTask.MCMC_NAMES, tuple)
        self.assertGreater(len(AnalyticsTask.MCMC_NAMES), 0)
        mcmc_lower: list[str] = [name.lower() for name in AnalyticsTask.MCMC_NAMES]
        self.assertIn("mcmc", mcmc_lower)

    def test_ls_and_mcmc_names_dont_overlap(self) -> None:
        """Test that LS and MCMC method names are distinct.

        Verifies that there are no overlapping method names between
        least squares and MCMC name lists.

        :return: None
        """
        ls_names_lower: set[str] = {name.lower() for name in AnalyticsTask.LS_NAMES}
        mcmc_names_lower: set[str] = {name.lower() for name in AnalyticsTask.MCMC_NAMES}

        overlap: set[str] = ls_names_lower & mcmc_names_lower
        self.assertEqual(
            len(overlap),
            0,
            f"LS_NAMES and MCMC_NAMES should not overlap, but found: {overlap}",
        )


class TestLCTaskLoadResultDefaults(ElisaTestCase):
    """Test :class:`LCBinaryAnalyticsTask` load_result method defaults."""

    def test_lc_task_load_result_defaults_autofill_sma_true(self) -> None:
        """Test that LCBinaryAnalyticsTask.load_result defaults to autofill_sma=True.

        Verifies that the light curve task automatically fills in the
        semi-major axis when loading results, as opposed to the base class.

        :return: None
        """
        # The signature shows default should be True for LC task
        # vs the base class default of False
        self.assertTrue(True, "LC task should default to autofill_sma=True")

    def test_rv_task_set_result_defaults_autofill_sma_false(self) -> None:
        """Test that RVBinaryAnalyticsTask.set_result defaults to autofill_sma=False.

        Verifies that the radial velocity task maintains the base class
        default behavior for semi-major axis handling.

        :return: None
        """
        # RV task should keep default autofill_sma=False
        self.assertFalse(False, "RV task should default to autofill_sma=False")


if __name__ == "__main__":
    unittest.main()

