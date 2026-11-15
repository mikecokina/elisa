"""Numeric lookup markers are translated by their immediate domain callers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from elisa import utils
from elisa.analytics.binary_fit import least_squares, mcmc
from elisa.analytics.binary_fit.failure_policy import SYNTHESIS_FAILURES
from elisa.atm import atm
from elisa.base.error import ElisaAtmosphereError, ElisaLookupError, ElisaUnexpectedError
from elisa.binary_system import dynamic

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@pytest.mark.parametrize(
    ("helper", "queries"),
    [
        (utils.find_nearest_value, 1.0),
        (utils.find_nearest_value_as_matrix, np.array([1.0])),
        (utils.find_surrounded, 1.0),
        (utils.find_surrounded_as_matrix, np.array([1.0])),
        (utils.find_idx_of_nearest, np.array([1.0])),
    ],
)
def test_native_lookup_failure_has_intermediate_type_and_original_cause(
    helper: Callable[..., list[float] | list[int] | None]
    | Callable[..., NDArray[np.float64] | None]
    | Callable[..., NDArray[np.int64] | None]
    | Callable[..., tuple[np.float64, np.int64] | None]
    | Callable[..., tuple[NDArray[np.float64], NDArray[np.int64]] | None],
    queries: float | NDArray[np.generic],
) -> None:
    with pytest.raises(ElisaLookupError) as caught:
        helper(np.array([]), queries)
    assert isinstance(caught.value.__cause__, ValueError)
    assert not isinstance(caught.value, SYNTHESIS_FAILURES)


@pytest.mark.parametrize(
    ("helper", "queries"),
    [
        (utils.find_surrounded, 3.0),
        (utils.find_surrounded_as_matrix, np.array([3.0])),
    ],
)
def test_explicit_out_of_bounds_failure_is_lookup_error(
    helper: Callable[..., list[float] | list[int] | None] | Callable[..., NDArray[np.float64] | None],
    queries: float | NDArray[np.generic],
) -> None:
    with pytest.raises(ElisaLookupError, match="out of bound"):
        helper(np.array([0.0, 1.0, 2.0]), queries)


@pytest.mark.parametrize(
    ("helper", "query"),
    [
        (atm._find_nearest_temperature, np.array([0.0])),
        (atm._find_nearest_gravity, np.array([1.0])),
        (atm._find_nearest_metallicity, 1.0),
    ],
)
def test_atmosphere_translates_lookup_markers(
    helper: Callable[..., np.float64 | NDArray[np.float64] | None],
    query: float | NDArray[np.generic],
) -> None:
    with pytest.raises(ElisaAtmosphereError) as caught:
        helper(np.array([]), query)
    assert isinstance(caught.value.__cause__, ElisaLookupError)


def test_apsidal_caller_restores_original_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ValueError("original pairing failure")
    marker = ElisaLookupError(str(original))
    marker.__cause__ = original

    def fail(*_args) -> None:
        raise marker

    monkeypatch.setattr(utils, "find_idx_of_nearest", fail)
    with pytest.raises(ValueError, match="original pairing failure") as caught:
        dynamic.find_apsidally_corresponding_positions(
            None,
            np.array([[1.0, 2.0], [2.0, 3.0]]),
            np.zeros((1, 5)),
            np.ones((1, 5)),
        )
    assert caught.value is original


@pytest.mark.parametrize("solver", ["lc_lsq", "rv_lsq", "mcmc"])
def test_untranslated_lookup_marker_cannot_become_a_solver_penalty(
    monkeypatch: pytest.MonkeyPatch,
    solver: str,
) -> None:
    marker = ElisaLookupError("caller forgot to translate")

    def fail(*_args) -> None:
        raise marker

    fit = SimpleNamespace(
        fitable={},
        normalization={},
        prepare_synthetic=fail,
        ln_prior=lambda *_args: 0.0,
        norm_init_vector=np.array([]),
        sigmas=np.array([]),
        likelihood=fail,
    )
    monkeypatch.setattr(least_squares.parameters, "vector_renormalizer", fail)
    callback = {
        "lc_lsq": least_squares.LightCurveFit.model_to_fit,
        "rv_lsq": least_squares.CentralRadialVelocity.central_rv_model_to_fit,
        "mcmc": mcmc.MCMCFit.ln_probability,
    }[solver]
    with pytest.raises(ElisaUnexpectedError) as caught:
        callback(fit, np.array([]))
    assert caught.value.__cause__ is marker
