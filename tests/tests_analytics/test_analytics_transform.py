"""Unit tests for elisa.analytics.transform module."""
from __future__ import annotations

import unittest
from typing import TYPE_CHECKING
from unittest import mock

from elisa import settings
from elisa.base.error import ElisaAnalyticsTaskError, ElisaPassbandError
from elisa.analytics.transform import (
    LCBinaryAnalyticsProperties,
    RVBinaryAnalyticsTask,
)
from tests import set_astropy_units
from tests.utils import ElisaTestCase

if TYPE_CHECKING:
    from elisa.analytics.dataset.base import LCData, RVData

set_astropy_units()


class TestRVBinaryAnalyticsTask(ElisaTestCase):
    """Test cases for :class:`RVBinaryAnalyticsTask` validator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        # Import here to avoid circular imports during module load
        from elisa.analytics.dataset.base import RVData

        self.RVData = RVData

    def test_data_valid_rv_dict(self) -> None:
        """Test validation of valid radial velocity data dictionary.

        Verifies that a properly formed RVData dictionary with valid
        binary counterpart keys is accepted and returned unchanged.

        :return: None
        """
        # Create mock RVData instances
        mock_rv_data: dict[str, RVData] = {}
        for counterpart in list(settings.BINARY_COUNTERPARTS.keys())[:2]:
            # Create a minimal mock RVData object
            mock_data = mock.MagicMock(spec=self.RVData)
            mock_rv_data[counterpart] = mock_data

        result = RVBinaryAnalyticsTask.data(mock_rv_data)
        self.assertEqual(result, mock_rv_data)
        self.assertIsInstance(result, dict)

    def test_data_not_dict_raises_task_error(self) -> None:
        """Test that non-dict input raises ElisaAnalyticsTaskError.

        Verifies that passing a non-dict value (like list, tuple, or string)
        raises a ElisaAnalyticsTaskError with appropriate message.

        :return: None
        :raises ElisaAnalyticsTaskError: When input is not a dict
        """
        invalid_inputs: list = [
            [],
            (),
            "not a dict",
            123,
            None,
            {"key": "value"},  # Will fail on isinstance check with RVData
        ]

        for invalid_input in invalid_inputs[:5]:  # Skip the dict for now
            with self.assertRaises(ElisaAnalyticsTaskError) as context:
                RVBinaryAnalyticsTask.data(invalid_input)
            self.assertIn("`radial_velocities`", str(context.exception))

    def test_data_invalid_counterpart_key(self) -> None:
        """Test that invalid binary counterpart key raises ElisaAnalyticsTaskError.

        Verifies that dictionary keys that are not valid binary counterpart
        designations raise a ElisaAnalyticsTaskError with helpful error message.

        :return: None
        :raises ElisaAnalyticsTaskError: When dict key is invalid counterpart designation
        """
        mock_data = mock.MagicMock(spec=self.RVData)
        invalid_data: dict = {"invalid_key": mock_data}

        with self.assertRaises(ElisaAnalyticsTaskError) as context:
            RVBinaryAnalyticsTask.data(invalid_data)

        error_msg: str = str(context.exception)
        self.assertIn("invalid_key", error_msg)
        self.assertIn("invalid designation", error_msg)

    def test_data_non_rvdata_instance_raises_task_error(self) -> None:
        """Test that non-RVData values raise ElisaAnalyticsTaskError.

        Verifies that dictionary values that are not RVData instances
        raise a ElisaAnalyticsTaskError with appropriate message.

        :return: None
        :raises ElisaAnalyticsTaskError: When dict value is not RVData instance
        """
        valid_key: str = list(settings.BINARY_COUNTERPARTS.keys())[0]
        invalid_data: dict = {valid_key: "not an RVData instance"}

        with self.assertRaises(ElisaAnalyticsTaskError) as context:
            RVBinaryAnalyticsTask.data(invalid_data)

        error_msg: str = str(context.exception)
        self.assertIn("not an instance of RVData", error_msg)

    def test_data_multiple_counterparts(self) -> None:
        """Test validation with multiple valid counterpart keys.

        Verifies that a dictionary with multiple valid binary counterpart
        keys and RVData instances is properly validated.

        :return: None
        """
        mock_data_dict: dict[str, RVData] = {}
        for counterpart in settings.BINARY_COUNTERPARTS.keys():
            mock_data = mock.MagicMock(spec=self.RVData)
            mock_data_dict[counterpart] = mock_data

        result = RVBinaryAnalyticsTask.data(mock_data_dict)
        self.assertEqual(len(result), len(settings.BINARY_COUNTERPARTS))
        self.assertIsInstance(result, dict)

    def test_data_empty_dict(self) -> None:
        """Test validation of empty dictionary.

        Verifies that an empty dictionary is accepted as valid input
        since it contains no invalid keys or values.

        :return: None
        """
        empty_dict: dict = {}
        result = RVBinaryAnalyticsTask.data(empty_dict)
        self.assertEqual(result, {})
        self.assertIsInstance(result, dict)


class TestLCBinaryAnalyticsProperties(ElisaTestCase):
    """Test cases for :class:`LCBinaryAnalyticsProperties` validator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        # Import here to avoid circular imports during module load
        from elisa.analytics.dataset.base import LCData

        self.LCData = LCData

    def test_data_valid_lc_dict(self) -> None:
        """Test validation of valid light curve data dictionary.

        Verifies that a properly formed LCData dictionary with valid
        passband keys is accepted and returned unchanged.

        :return: None
        """
        # Create mock LCData instances
        mock_lc_data: dict[str, LCData] = {}
        for passband in list(settings.PASSBANDS)[:2]:
            mock_data = mock.MagicMock(spec=self.LCData)
            mock_lc_data[passband] = mock_data

        result = LCBinaryAnalyticsProperties.data(mock_lc_data)
        self.assertEqual(result, mock_lc_data)
        self.assertIsInstance(result, dict)

    def test_data_not_dict_raises_task_error(self) -> None:
        """Test that non-dict input raises ElisaAnalyticsTaskError.

        Verifies that passing a non-dict value (like list, tuple, or string)
        raises a ElisaAnalyticsTaskError with appropriate message for light curves.

        :return: None
        :raises ElisaAnalyticsTaskError: When input is not a dict
        """
        invalid_inputs: list = [
            [],
            (),
            "not a dict",
            123,
            None,
        ]

        for invalid_input in invalid_inputs:
            with self.assertRaises(ElisaAnalyticsTaskError) as context:
                LCBinaryAnalyticsProperties.data(invalid_input)
            self.assertIn("`light_curves`", str(context.exception))

    def test_data_invalid_passband_key(self) -> None:
        """Test that invalid passband key raises ElisaPassbandError.

        Verifies that dictionary keys that are not valid passband
        designations raise a ElisaPassbandError with helpful error message listing
        available passbands.

        :return: None
        :raises ElisaPassbandError: When dict key is invalid passband designation
        """
        mock_data = mock.MagicMock(spec=self.LCData)
        invalid_data: dict = {"invalid_passband": mock_data}

        with self.assertRaises(ElisaPassbandError) as context:
            LCBinaryAnalyticsProperties.data(invalid_data)

        error_msg: str = str(context.exception)
        self.assertIn("invalid_passband", error_msg)
        self.assertIn("invalid passband", error_msg)

    def test_data_non_lcdata_instance_raises_task_error(self) -> None:
        """Test that non-LCData values raise ElisaAnalyticsTaskError.

        Verifies that dictionary values that are not LCData instances
        raise a ElisaAnalyticsTaskError with appropriate message.

        :return: None
        :raises ElisaAnalyticsTaskError: When dict value is not LCData instance
        """
        valid_passband: str = list(settings.PASSBANDS)[0]
        invalid_data: dict = {valid_passband: "not an LCData instance"}

        with self.assertRaises(ElisaAnalyticsTaskError) as context:
            LCBinaryAnalyticsProperties.data(invalid_data)

        error_msg: str = str(context.exception)
        self.assertIn("not an instance of LCData", error_msg)

    def test_data_multiple_passbands(self) -> None:
        """Test validation with multiple valid passband keys.

        Verifies that a dictionary with multiple valid passband keys and
        LCData instances is properly validated.

        :return: None
        """
        mock_data_dict: dict[str, LCData] = {}
        for passband in list(settings.PASSBANDS)[:3]:
            mock_data = mock.MagicMock(spec=self.LCData)
            mock_data_dict[passband] = mock_data

        result = LCBinaryAnalyticsProperties.data(mock_data_dict)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result, dict)

    def test_data_empty_dict(self) -> None:
        """Test validation of empty dictionary.

        Verifies that an empty dictionary is accepted as valid input
        since it contains no invalid keys or values.

        :return: None
        """
        empty_dict: dict = {}
        result = LCBinaryAnalyticsProperties.data(empty_dict)
        self.assertEqual(result, {})
        self.assertIsInstance(result, dict)

    def test_data_all_passbands(self) -> None:
        """Test validation with all available passbands.

        Verifies that a dictionary with all available passbands from
        settings is properly validated.

        :return: None
        """
        mock_data_dict: dict[str, LCData] = {}
        for passband in settings.PASSBANDS:
            mock_data = mock.MagicMock(spec=self.LCData)
            mock_data_dict[passband] = mock_data

        result = LCBinaryAnalyticsProperties.data(mock_data_dict)
        self.assertEqual(len(result), len(settings.PASSBANDS))
        self.assertIsInstance(result, dict)


class TestRVBinaryAnalyticsTaskEdgeCases(ElisaTestCase):
    """Test edge cases and error conditions for RVBinaryAnalyticsTask."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        from elisa.analytics.dataset.base import RVData

        self.RVData = RVData

    def test_data_mixed_valid_and_invalid_keys(self) -> None:
        """Test dict with both valid and invalid keys raises ElisaAnalyticsTaskError.

        Verifies that even if most keys are valid, presence of a single
        invalid key causes the validation to fail.

        :return: None
        :raises ElisaAnalyticsTaskError: When any dict key is invalid
        """
        mock_data = mock.MagicMock(spec=self.RVData)
        valid_key: str = list(settings.BINARY_COUNTERPARTS.keys())[0]

        mixed_data: dict = {
            valid_key: mock_data,
            "invalid_key": mock_data,
        }

        with self.assertRaises(ElisaAnalyticsTaskError):
            RVBinaryAnalyticsTask.data(mixed_data)

    def test_data_none_value_in_dict(self) -> None:
        """Test that None value in dict raises ElisaAnalyticsTaskError.

        Verifies that None values (even with valid keys) are rejected
        as not being RVData instances.

        :return: None
        :raises ElisaAnalyticsTaskError: When dict value is None
        """
        valid_key: str = list(settings.BINARY_COUNTERPARTS.keys())[0]
        invalid_data: dict = {valid_key: None}

        with self.assertRaises(ElisaAnalyticsTaskError):
            RVBinaryAnalyticsTask.data(invalid_data)


class TestLCBinaryAnalyticsPropertiesEdgeCases(ElisaTestCase):
    """Test edge cases and error conditions for LCBinaryAnalyticsProperties."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        from elisa.analytics.dataset.base import LCData

        self.LCData = LCData

    def test_data_mixed_valid_and_invalid_passbands(self) -> None:
        """Test dict with both valid and invalid passbands raises ElisaPassbandError.

        Verifies that even if most passbands are valid, presence of a single
        invalid passband causes the validation to fail.

        :return: None
        :raises ElisaPassbandError: When any dict key is invalid passband
        """
        mock_data = mock.MagicMock(spec=self.LCData)
        valid_passband: str = list(settings.PASSBANDS)[0]

        mixed_data: dict = {
            valid_passband: mock_data,
            "invalid_passband": mock_data,
        }

        with self.assertRaises(ElisaPassbandError):
            LCBinaryAnalyticsProperties.data(mixed_data)

    def test_data_none_value_in_dict(self) -> None:
        """Test that None value in dict raises ElisaAnalyticsTaskError.

        Verifies that None values (even with valid passband keys) are
        rejected as not being LCData instances.

        :return: None
        :raises ElisaAnalyticsTaskError: When dict value is None
        """
        valid_passband: str = list(settings.PASSBANDS)[0]
        invalid_data: dict = {valid_passband: None}

        with self.assertRaises(ElisaAnalyticsTaskError):
            LCBinaryAnalyticsProperties.data(invalid_data)

    def test_error_message_includes_available_passbands(self) -> None:
        """Test that error message includes list of available passbands.

        Verifies that when an invalid passband is provided, the error
        message includes helpful information about available passbands.

        :return: None
        """
        mock_data = mock.MagicMock(spec=self.LCData)
        invalid_data: dict = {"invalid_passband": mock_data}

        with self.assertRaises(ElisaPassbandError) as context:
            LCBinaryAnalyticsProperties.data(invalid_data)

        error_msg: str = str(context.exception)
        # Should mention available passbands in some form
        self.assertIn("available", error_msg.lower())


if __name__ == "__main__":
    unittest.main()

