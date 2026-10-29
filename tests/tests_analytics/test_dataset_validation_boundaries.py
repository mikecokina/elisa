"""Dataset input failures are normalized at numeric and unit conversion boundaries."""
from unittest.mock import patch

import numpy as np
import pytest

from elisa import units as u
from elisa.analytics.dataset import utils as dutils
from elisa.analytics.dataset.base import LCData, RVData
from elisa.base.error import (
    ElisaAnalyticsDataValidationError,
    ElisaInvalidKwargsError,
    ElisaMissingKwargsError,
)


@pytest.fixture(params=[RVData, LCData])
def dataset_case(request):
    dataset = request.param
    kwargs = dict(
        x_data=[0.0, 0.5], y_data=[1.0, 2.0], y_err=[0.1, 0.2],
        x_unit=u.dimensionless_unscaled,
        y_unit=u.m / u.s if dataset is RVData else u.dimensionless_unscaled,
    )
    return dataset, kwargs


@pytest.mark.parametrize("field", ["x_data", "y_data", "y_err"])
@pytest.mark.parametrize("values", [["abc", "def"], [object(), object()], [[1.0], [2.0, 3.0]]])
def test_invalid_numeric_fields_report_field_and_original_cause(dataset_case, field, values):
    dataset, kwargs = dataset_case
    kwargs[field] = values
    with pytest.raises(ElisaAnalyticsDataValidationError, match=field) as caught:
        dataset(**kwargs)
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))


@pytest.mark.parametrize("field", ["x_unit", "y_unit"])
@pytest.mark.parametrize("unit", ["invalid", object(), 123])
def test_invalid_unit_objects_report_property_and_original_cause(dataset_case, field, unit):
    dataset, kwargs = dataset_case
    kwargs[field] = unit
    with pytest.raises(ElisaAnalyticsDataValidationError, match=field) as caught:
        dataset(**kwargs)
    assert isinstance(caught.value.__cause__, (TypeError, ValueError, AttributeError))


@pytest.mark.parametrize("stage", ["transform_input", "convert_arrays"])
@pytest.mark.parametrize("error_type", [TypeError, ValueError, AttributeError, FloatingPointError, OverflowError])
def test_later_processing_failures_use_data_validation_error(dataset_case, stage, error_type):
    dataset, kwargs = dataset_case
    # Exercise the actual dispatch and conversion boundaries without changing
    # the initial validation of otherwise valid input arrays.
    from elisa.analytics.dataset.transform import RVDataProperties, LCDataProperties

    failure = error_type("invalid input representation")
    target = (RVDataProperties if dataset is RVData else LCDataProperties) if stage == "transform_input" else dutils
    method = "x_unit" if stage == "transform_input" else "convert_data"
    with patch.object(target, method, side_effect=failure):
        with pytest.raises(ElisaAnalyticsDataValidationError) as caught:
            dataset(**kwargs)
    assert caught.value.__cause__ is failure
    assert ("x_unit" if stage == "transform_input" else dataset.__name__) in str(caught.value)


@pytest.mark.parametrize("zero_point", ["invalid", object(), [1.0, 2.0, 3.0], 1 * u.m])
def test_bad_magnitude_reference_is_normalized_during_conversion(zero_point):
    with pytest.raises(ElisaAnalyticsDataValidationError, match="LCData") as caught:
        LCData(x_data=[0.0, 0.5], y_data=[10.0, 11.0], y_err=[0.1, 0.1],
               x_unit=u.dimensionless_unscaled, y_unit=u.mag, reference_magnitude=zero_point)
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))


@pytest.mark.parametrize("stage", ["transform_input", "convert_arrays"])
def test_unexpected_runtime_errors_are_not_reclassified(dataset_case, stage):
    dataset, kwargs = dataset_case
    failure = RuntimeError("unexpected programming failure")
    from elisa.analytics.dataset.transform import RVDataProperties, LCDataProperties

    target = (RVDataProperties if dataset is RVData else LCDataProperties) if stage == "transform_input" else dutils
    method = "x_unit" if stage == "transform_input" else "convert_data"
    with patch.object(target, method, side_effect=failure):
        with pytest.raises(RuntimeError) as caught:
            dataset(**kwargs)
    assert caught.value is failure


def test_unknown_and_missing_kwargs_keep_their_dedicated_errors(dataset_case):
    dataset, kwargs = dataset_case
    with pytest.raises(ElisaInvalidKwargsError):
        dataset(**kwargs, unknown=1)
    del kwargs["x_data"]
    with pytest.raises(ElisaMissingKwargsError):
        dataset(**kwargs)


def test_supplied_validation_errors_are_not_wrapped_again(dataset_case):
    dataset, kwargs = dataset_case
    failure = ElisaAnalyticsDataValidationError("specific conversion diagnostics")
    with patch.object(dutils, "convert_data", side_effect=failure):
        with pytest.raises(ElisaAnalyticsDataValidationError) as caught:
            dataset(**kwargs)
    assert caught.value is failure


def test_valid_rv_unit_conversions_and_optional_errors_are_preserved():
    data = RVData(x_data=[0.0, 24.0], y_data=[1.0, 2.0], x_unit=u.Unit("h"), y_unit=u.km / u.s)
    np.testing.assert_allclose(data.x_data, [0.0, 1.0])
    np.testing.assert_allclose(data.y_data, [1000.0, 2000.0])
    assert data.y_err is None


def test_valid_magnitude_conversion_and_uncertainties_are_preserved():
    data = LCData(x_data=[0.0, 0.5], y_data=[10.0, 11.0], y_err=[0.1, 0.2],
                  x_unit=u.dimensionless_unscaled, y_unit=u.mag, reference_magnitude=10.0)
    flux = 10.0 ** ((10.0 - np.array([10.0, 11.0])) / 2.5)
    np.testing.assert_allclose(data.y_data, flux)
    np.testing.assert_allclose(data.y_err, (np.log(10.0) / 2.5) * flux * [0.1, 0.2])


def test_nonnumeric_uncertainties_are_normalized_during_positivity_check(dataset_case):
    dataset, kwargs = dataset_case
    kwargs["y_err"] = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]")
    with pytest.raises(ElisaAnalyticsDataValidationError, match="y_err") as caught:
        dataset(**kwargs)
    assert isinstance(caught.value.__cause__, TypeError)


def test_direct_array_conversion_normalizes_unit_failures(dataset_case):
    from astropy.units import UnitConversionError

    dataset, kwargs = dataset_case
    kwargs["x_data"] = np.asarray(kwargs["x_data"])
    kwargs["x_unit"] = u.m
    with pytest.raises(ElisaAnalyticsDataValidationError, match=dataset.__name__) as caught:
        dataset.convert_arrays(**kwargs)
    assert isinstance(caught.value.__cause__, UnitConversionError)


def test_property_transformation_handles_empty_input_and_first_property_failure(dataset_case):
    dataset, kwargs = dataset_case
    data = dataset(**kwargs)
    assert data.transform_input() == {}
    with pytest.raises(ElisaAnalyticsDataValidationError, match="`x_unit`") as caught:
        data.transform_input(x_unit=object())
    assert isinstance(caught.value.__cause__, AttributeError)
