from __future__ import annotations

import pytest
# noinspection PyProtectedMember
from elisa.conf.parsers import _parse_number, parse_tuple_interval


def test_parse_number_int():
    """Test _parse_number returns int for integer string."""
    assert _parse_number("42") == 42
    assert isinstance(_parse_number("42"), int)


def test_parse_number_float():
    """Test _parse_number returns float for float string."""
    assert _parse_number("3.14") == 3.14
    assert isinstance(_parse_number("3.14"), float)
    assert _parse_number("2e2") == 200.0
    assert isinstance(_parse_number("2e2"), float)


def test_parse_tuple_interval_ints():
    """Test parse_tuple_interval with two integers."""
    result = parse_tuple_interval("(1, 2)", name="test")
    assert result == (1, 2)
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)


def test_parse_tuple_interval_floats():
    """Test parse_tuple_interval with floats and scientific notation."""
    result = parse_tuple_interval("(1.5, 2.5)", name="test")
    assert result == (1.5, 2.5)
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    result = parse_tuple_interval("(1e2, 2e2)", name="test")
    assert result == (100.0, 200.0)


def test_parse_tuple_interval_mixed():
    """Test parse_tuple_interval with mixed int and float."""
    result = parse_tuple_interval("(1, 2.5)", name="test")
    assert result == (1, 2.5)
    assert isinstance(result[0], int)
    assert isinstance(result[1], float)


def test_parse_tuple_interval_spaces():
    """Test parse_tuple_interval with extra spaces."""
    result = parse_tuple_interval(" (  3 ,  4.0 ) ", name="test")
    assert result == (3, 4.0)


def test_parse_tuple_interval_ordered():
    """Test parse_tuple_interval require_ordered=True (default)."""
    result = parse_tuple_interval("(1, 2)", name="test")
    assert result == (1, 2)
    with pytest.raises(ValueError, match="low > high"):
        parse_tuple_interval("(2, 1)", name="test")


def test_parse_tuple_interval_not_ordered():
    """Test parse_tuple_interval require_ordered=False."""
    result = parse_tuple_interval("(2, 1)", name="test", require_ordered=False)
    assert result == (2, 1)


def test_parse_tuple_interval_invalid_format():
    """Test parse_tuple_interval with invalid format raises ValueError."""
    with pytest.raises(ValueError, match="invalid format"):
        parse_tuple_interval("1, 2", name="test")
    with pytest.raises(ValueError, match="invalid format"):
        parse_tuple_interval("(1 2)", name="test")
    with pytest.raises(ValueError, match="invalid format"):
        parse_tuple_interval("(a, b)", name="test")


def test_parse_tuple_interval_missing():
    """Test parse_tuple_interval with None input raises ValueError."""
    with pytest.raises(ValueError, match="missing value"):
        parse_tuple_interval(None, name="test")
