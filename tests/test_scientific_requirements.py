"""Verify scientific data requirements using read-only, explicit file fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from elisa import ld
from elisa.atm import atm
from elisa.base.error import ElisaAtmosphereError
from tests.scientific_requirements import (
    atmosphere_unavailability_reason,
    ld_unavailability_reason,
)

if TYPE_CHECKING:
    from typing import NoReturn

TEST_DATA_DIRECTORY = Path(__file__).parent / "data"
CK04_FIXTURE_DIRECTORY = TEST_DATA_DIRECTORY / "ck04"
CK04_FIXTURE_MODELS = (
    "ckp00/ckp00_10000_g40.csv",
    "ckp00/ckp00_10250_g40.csv",
)


@pytest.mark.parametrize(
    ("passband", "law", "metallicity"),
    [
        pytest.param("Generic.Bessell.B", "linear", -0.2, id="linear-bessell-b"),
        pytest.param("Kepler", "logarithmic", -4.0, id="logarithmic-kepler"),
    ],
)
def test_available_van_hamme_table_is_accepted(passband: str, law: str, metallicity: float) -> None:
    reason = ld_unavailability_reason(
        TEST_DATA_DIRECTORY / "vh93",
        passbands=(passband,),
        laws=(law,),
        metallicities=(metallicity,),
        source="van_hamme",
    )

    assert reason is None


def test_ld_requirement_distinguishes_laws_for_the_same_passband() -> None:
    reason = ld_unavailability_reason(
        TEST_DATA_DIRECTORY / "vh93",
        passbands=("Kepler",),
        laws=("linear",),
        metallicities=(-4.0,),
    )

    assert reason is not None
    assert "lin.Kepler.m40.csv" in reason


def test_ld_requirement_accepts_both_metallicity_neighbours() -> None:
    reason = ld_unavailability_reason(
        TEST_DATA_DIRECTORY / "ld_grid",
        passbands=("Generic.Bessell.B",),
        laws=("linear",),
        metallicities=(-0.15,),
    )

    assert reason is None


def test_ld_requirement_reports_a_missing_metallicity_neighbour() -> None:
    reason = ld_unavailability_reason(
        TEST_DATA_DIRECTORY / "ld_grid",
        passbands=("Generic.Bessell.B",),
        laws=("linear",),
        metallicities=(-0.05,),
    )

    assert reason is not None
    assert "lin.Generic.Bessell.B.p00.csv" in reason


@pytest.mark.parametrize(
    "table_directory",
    [
        pytest.param(None, id="unset"),
        pytest.param("", id="empty"),
        pytest.param(" ", id="blank"),
    ],
)
def test_unconfigured_ld_directory_has_a_clear_reason(table_directory: str | None) -> None:
    reason = ld_unavailability_reason(table_directory)

    assert reason == "Van Hamme LD tables (LD_TABLES) is not configured"


def test_missing_ld_directory_is_unavailable(tmp_path: Path) -> None:
    missing_directory = tmp_path / "missing-installation"

    reason = ld_unavailability_reason(missing_directory)

    assert reason is not None
    assert str(missing_directory) in reason
    assert "is not a directory" in reason


def test_empty_ld_directory_is_unavailable(tmp_path: Path) -> None:
    reason = ld_unavailability_reason(tmp_path)

    assert reason is not None
    assert "missing or empty required file" in reason


def test_empty_atmosphere_directory_is_unavailable(tmp_path: Path) -> None:
    reason = atmosphere_unavailability_reason(tmp_path, atlas="ck04")

    assert reason is not None
    assert "no model CSV files" in reason


def test_atmosphere_requirement_accepts_exact_fixture_models() -> None:
    reason = atmosphere_unavailability_reason(CK04_FIXTURE_DIRECTORY, atlas="ck04", files=CK04_FIXTURE_MODELS)

    assert reason is None


def test_atmosphere_requirement_reports_an_incomplete_grid() -> None:
    missing_model = "ckp00/ckp00_5000_g40.csv"

    reason = atmosphere_unavailability_reason(
        CK04_FIXTURE_DIRECTORY,
        atlas="ck04",
        files=(*CK04_FIXTURE_MODELS, missing_model),
    )

    assert reason is not None
    assert missing_model in reason


@pytest.mark.parametrize("atlas", ["ck04", "castelli"])
def test_atmosphere_requirement_accepts_atlas_aliases(atlas: str) -> None:
    reason = atmosphere_unavailability_reason(CK04_FIXTURE_DIRECTORY, atlas=atlas)

    assert reason is None


@pytest.mark.parametrize("atlas", ["bb", "black_body"])
def test_black_body_requires_no_installed_atmosphere(atlas: str) -> None:
    reason = atmosphere_unavailability_reason(None, atlas=atlas)

    assert reason is None


def test_missing_k93_models_are_reported_as_k93(tmp_path: Path) -> None:
    reason = atmosphere_unavailability_reason(tmp_path, atlas="k93")

    assert reason is not None
    assert "k93 atmosphere models (K93_ATM_TABLES)" in reason
    assert "no model CSV files" in reason


def test_unknown_atlas_has_a_clear_reason() -> None:
    reason = atmosphere_unavailability_reason(CK04_FIXTURE_DIRECTORY, atlas="invalid-atlas")

    assert reason == "Atmosphere atlas 'invalid-atlas' is not supported"


def test_availability_checks_do_not_parse_or_synthesize(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbid_scientific_loading(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("Availability checks must not parse or synthesize scientific data")

    monkeypatch.setattr(ld, "get_ld_source_interpolator", forbid_scientific_loading)
    monkeypatch.setattr(atm, "get_atm_table", forbid_scientific_loading)
    monkeypatch.setattr(atm.NaiveInterpolatedAtm, "atlas_radiance", forbid_scientific_loading)

    ld_reason = ld_unavailability_reason(
        TEST_DATA_DIRECTORY / "vh93",
        passbands=("Kepler",),
        laws=("logarithmic",),
        metallicities=(-4.0,),
    )
    atmosphere_reason = atmosphere_unavailability_reason(CK04_FIXTURE_DIRECTORY, atlas="ck04")

    assert ld_reason is None
    assert atmosphere_reason is None


def test_missing_model_after_successful_preflight_raises(tmp_path: Path) -> None:
    reason = atmosphere_unavailability_reason(CK04_FIXTURE_DIRECTORY, atlas="ck04")
    assert reason is None

    missing_model = tmp_path / "ckp00_5000_g40.csv"
    with pytest.raises(ElisaAtmosphereError, match="doesn't exist") as raised:
        atm.read_unique_atm_tables([str(missing_model)])

    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_malformed_model_after_successful_preflight_raises(tmp_path: Path) -> None:
    source_model = CK04_FIXTURE_DIRECTORY / CK04_FIXTURE_MODELS[0]
    model_directory = tmp_path / "ckp00"
    model_directory.mkdir()
    malformed_model = model_directory / source_model.name
    contents = source_model.read_text(encoding="utf-8")
    malformed_model.write_text(contents.replace("flux,wave", "invalid_flux,wave", 1), encoding="utf-8")

    reason = atmosphere_unavailability_reason(tmp_path, atlas="ck04")
    assert reason is None

    with pytest.raises(KeyError, match="flux"):
        atm.read_unique_atm_tables([str(malformed_model)])
