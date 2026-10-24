"""Scientific table configuration shared by the real synthesis tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from elisa import atm, ld, settings
from elisa.analytics.binary_fit.lc_fit import LCFitMCMC
from elisa.analytics.binary_fit.rv_fit import RVFitMCMC
from elisa.buffer import buffer


@pytest.fixture
def science_settings() -> Iterator[None]:
    support = Path(os.environ.get("ELISA_SCIENCE_SUPPORT", Path.home() / ".elisa"))
    ld_tables = support / "limbdarkening" / "ld"
    atmosphere = support / "atmosphere" / "ck04"
    if not (ld_tables / "lin.Generic.Bessell.V.p00.csv").is_file() or not (atmosphere / "ckp00").is_dir():
        pytest.skip("Real ELISA LD/CK04 tables required; set ELISA_SCIENCE_SUPPORT")
    previous = settings.settings_serializer()
    settings.configure(
        LD_TABLES=str(ld_tables),
        CK04_ATM_TABLES=str(atmosphere),
        ATM_ATLAS="bb",
        LIMB_DARKENING_LAW="linear",
        USE_SINGLE_LD_COEFFICIENTS=True,
        NUMBER_OF_PROCESSES=1,
        MESH_GENERATOR="auto",
        REFLECTION_EFFECT=True,
        REFLECTION_EFFECT_ITERATIONS=2,
        MAX_CURVE_DATAPOINTS=12,
    )
    try:
        yield
    finally:
        settings.configure(**previous)
        ld.clear_ld_caches()
        buffer.ATMOSPHERE_TABLES.clear()
        atm.clear_atm_strip_cache()


@pytest.fixture(params=["lc", "rv"])
def mcmc_result(request: pytest.FixtureRequest) -> LCFitMCMC | RVFitMCMC:
    if request.param == "lc":
        return LCFitMCMC("detached", None, None)
    return RVFitMCMC()
