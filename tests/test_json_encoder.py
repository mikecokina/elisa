"""JSON representations of supported scientific types and strict fallbacks."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy import units as u

from elisa.json_encoder import ElisaJSONEncoder

if TYPE_CHECKING:
    import datetime as dt

    import astropy.units.core
    import astropy.units.quantity
    from numpy.typing import NDArray


def assert_encoded_value(
    value: date
    | dt
    | NDArray[np.bool]
    | NDArray[np.float32]
    | NDArray[np.float64]
    | NDArray[np.int64]
    | NDArray[np.object_]
    | NDArray[np.str_],
    expected: int | list[bool] | list[float] | list[list[int]] | list[object] | list[str] | str,
) -> None:
    assert json.loads(json.dumps(value, cls=ElisaJSONEncoder)) == expected


@pytest.mark.parametrize(
    ("value", "expected", "native_type"),
    [
        *[(kind(-7), -7, int) for kind in (np.int8, np.int16, np.int32, np.int64)],
        *[(kind(7), 7, int) for kind in (np.uint8, np.uint16, np.uint32, np.uint64)],
        (np.uint64(2**64 - 1), 2**64 - 1, int),
        *[(kind(1.25), 1.25, float) for kind in (np.float16, np.float32, np.float64, np.longdouble)],
        (np.bool_(True), True, bool),
        (np.bool_(False), False, bool),
    ],
)
def test_numpy_scalars(
    value: np.bool
    | np.float16
    | np.float32
    | np.float64
    | np.int16
    | np.int32
    | np.int64
    | np.int8
    | np.longdouble
    | np.uint16
    | np.uint32
    | np.uint64
    | np.uint8,
    expected: bool | float,
    native_type: type[bool | float | int],
) -> None:
    decoded = json.loads(json.dumps({"value": value}, cls=ElisaJSONEncoder))["value"]
    assert decoded == expected
    assert type(decoded) is native_type


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.array([[1, 2], [3, 4]], dtype=np.int64), [[1, 2], [3, 4]]),
        (np.array([1.25, 2.5], dtype=np.float32), [1.25, 2.5]),
        (np.array([True, False]), [True, False]),
        (np.array(["a", "b"]), ["a", "b"]),
        (np.array([], dtype=float), []),
        (np.array(3), 3),
        (np.array([u.K, date(2026, 10, 28)], dtype=object), ["K", "2026-10-28"]),
    ],
)
def test_numpy_arrays(
    value: NDArray[np.generic],
    expected: int | list[bool] | list[float] | list[list[int]] | list[object] | list[str],
) -> None:
    assert_encoded_value(value, expected)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (date(2026, 10, 28), "2026-10-28"),
        (
            datetime(2026, 10, 28, 12, 34, 56, 123456),  # noqa: DTZ001 - explicitly test naive datetime serialization
            "2026-10-28T12:34:56.123456",
        ),
        (datetime(2026, 10, 28, tzinfo=UTC), "2026-10-28T00:00:00+00:00"),
        (
            datetime(2026, 10, 28, 12, tzinfo=timezone(timedelta(hours=1))),
            "2026-10-28T12:00:00+01:00",
        ),
    ],
)
def test_iso_dates(value: dt.date | dt.datetime, expected: str) -> None:
    assert_encoded_value(value, expected)


@pytest.mark.parametrize("unit", ["K", "m/s", " km / s ", "solRad", "", None, u.km / u.s])
def test_values_and_unit_strings_are_preserved(unit: astropy.units.core.CompositeUnit | str | None) -> None:
    original = {"value": 42.125, "unit": unit}
    decoded = json.loads(json.dumps(original, cls=ElisaJSONEncoder))
    assert decoded == {"value": 42.125, "unit": str(unit) if isinstance(unit, u.UnitBase) else unit}
    assert type(decoded["value"]) is float
    assert original["unit"] is unit


class Stringable:
    def __str__(self) -> str:
        return "must not be serialized"


@pytest.mark.parametrize(
    "value",
    [
        object(),
        Stringable(),
        {1, 2},
        1 + 2j,
        np.complex128(1 + 2j),
        3 * u.K,
        np.array([1, 2]) * u.km,
        3 * u.dimensionless_unscaled,
        np.array([1, 2]) * u.dimensionless_unscaled,
        np.array([Stringable()], dtype=object),
    ],
)
def test_unsupported_types_raise_type_error(
    value: astropy.units.quantity.Quantity
    | complex
    | np.complex128
    | NDArray[np.generic]
    | object
    | set[int]
    | Stringable,
) -> None:
    with pytest.raises(TypeError, match="is not JSON serializable"):
        json.dumps({"value": value}, cls=ElisaJSONEncoder)
