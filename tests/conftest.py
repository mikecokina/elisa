"""Shared pytest options and installed scientific data requirements."""

import pytest

from tests.scientific_requirements import require_scientific_data


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register opt-in browser fitting tests."""
    parser.addoption(
        "--run-fitting-e2e",
        action="store_true",
        help="Run real scientific browser fits",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the scientific data markers accepted by the test suite."""
    config.addinivalue_line(
        "markers",
        'requires_ld(passbands, laws, metallicities, source="van_hamme"): '
        "requires configured LD tables; law/metal/band parameters are resolved per case",
    )
    config.addinivalue_line(
        "markers",
        "requires_atmosphere(atlas=None, files=None): requires configured atmosphere models",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Check installed data before any test-specific setup runs."""
    require_scientific_data(item)
