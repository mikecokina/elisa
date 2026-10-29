"""Dataset input failures are normalized at numeric and unit conversion boundaries."""
from unittest.mock import patch

import numpy as np
import pandas as pd
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


@pytest.fixture(params=[None, RVData, LCData], ids=["reader", "RVData", "LCData"])
def observation_file_loader(request):
    def load(path, *, data_columns=(0, 1, 2), delimiter=r"\s+"):
        if request.param is None:
            return dutils.read_data_file(str(path), data_columns, delimiter=delimiter)
        return request.param.load_from_file(
            str(path), data_columns=data_columns, delimiter=delimiter,
            x_unit=u.dimensionless_unscaled,
            y_unit=u.m / u.s if request.param is RVData else u.dimensionless_unscaled,
        )
    return load


@pytest.mark.parametrize("content,cause,message", [
    (b"", pd.errors.EmptyDataError, "Cannot parse"),
    (b"# comments only\n", pd.errors.EmptyDataError, "Cannot parse"),
    (b"0 1 .1\n1 2 .2 extra\n", pd.errors.ParserError, "Cannot parse"),
    (b"\xff\xfe\n", UnicodeDecodeError, "Cannot parse"),
    (b"0 abc .1\n", ValueError, "Invalid numeric"),
    (b"0 1\n", KeyError, "Cannot select"),
    (b"0 1 inf\n", None, "nonfinite"),
    (b"0 1 .1\n1 2\n", None, "nonfinite"),
])
def test_observation_file_validation_reports_file_and_original_cause(
    tmp_path, observation_file_loader, content, cause, message,
):
    path = tmp_path / "observations.dat"
    path.write_bytes(content)
    with pytest.raises(ElisaAnalyticsDataValidationError, match=message) as caught:
        observation_file_loader(path)
    assert str(path) in str(caught.value)
    if cause is None:
        assert caught.value.__cause__ is None
    else:
        assert isinstance(caught.value.__cause__, cause)
        assert str(caught.value.__cause__) in str(caught.value)
    if message != "Cannot parse":
        assert "(0, 1, 2)" in str(caught.value)


@pytest.mark.parametrize("content,columns,delimiter,expected", [
    ("# header\n1 0 1 .1 unused\n2 .5 .9 .2 ignored\n", (1, 2, 3), r"\s+",
     [[0.0, 1.0, 0.1], [0.5, 0.9, 0.2]]),
    ("unused,0,1\nignored,.5,.9\n", (1, 2), ",", [[0.0, 1.0], [0.5, 0.9]]),
])
def test_valid_observation_files_preserve_column_selection_and_optional_errors(
    tmp_path, observation_file_loader, content, columns, delimiter, expected,
):
    path = tmp_path / "observations.dat"
    path.write_text(content)
    result = observation_file_loader(path, data_columns=columns, delimiter=delimiter)
    expected = np.asarray(expected)
    if isinstance(result, np.ndarray):
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float64
    else:
        np.testing.assert_array_equal(result.x_data, expected[:, 0])
        np.testing.assert_array_equal(result.y_data, expected[:, 1])
        if len(columns) == 3:
            np.testing.assert_array_equal(result.y_err, expected[:, 2])
        else:
            assert result.y_err is None


def test_observation_file_io_errors_keep_their_type(tmp_path, observation_file_loader):
    with pytest.raises(FileNotFoundError):
        observation_file_loader(tmp_path / "missing.dat")


@pytest.mark.parametrize("columns,cause", [((0, 99), KeyError), (1, TypeError)])
def test_invalid_observation_column_selection_uses_data_error(tmp_path, observation_file_loader, columns, cause):
    path = tmp_path / "observations.dat"
    path.write_text("0 1 .1\n")
    with pytest.raises(ElisaAnalyticsDataValidationError, match="Cannot select") as caught:
        observation_file_loader(path, data_columns=columns)
    assert isinstance(caught.value.__cause__, cause)
    assert repr(columns) in str(caught.value)
    assert str(path) in str(caught.value)


def test_unexpected_reader_runtime_error_is_not_reclassified(tmp_path):
    failure = RuntimeError("unexpected reader failure")
    with patch.object(dutils.pd, "read_csv", side_effect=failure):
        with pytest.raises(RuntimeError) as caught:
            dutils.read_data_file(str(tmp_path / "observations.dat"), (0, 1, 2))
    assert caught.value is failure
