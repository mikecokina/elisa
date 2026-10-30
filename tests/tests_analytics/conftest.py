"""Numerical synthesis controls; installed data requirements use pytest markers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from elisa import ld, settings
from elisa.atm import atm
from elisa.analytics.binary_fit.lc_fit import LCFitMCMC
from elisa.analytics.binary_fit.rv_fit import RVFitMCMC
from elisa.buffer import buffer

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def science_settings() -> Iterator[None]:
    previous = settings.settings_serializer()
    settings.configure(
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
