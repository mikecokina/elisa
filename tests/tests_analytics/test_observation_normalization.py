"""Light-curve normalization and smoothing preserve observational weights."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elisa import units as u
from elisa.analytics import LCData
from elisa.analytics.binary_fit import least_squares, shared
from elisa.observer.utils import normalize_light_curve
from tests.tests_analytics.fitting_helpers import V, lc_setup, make_lc


@pytest.mark.parametrize(
    ("cyclic", "errors", "expected_flux", "expected_errors"),
    [
        pytest.param(False, None, [14 / 3, 7.5, 56 / 3, 24.0], None, id="open-unweighted"),
        pytest.param(True, None, [11.5, 7.5, 56 / 3, 13.5], None, id="cyclic-unweighted"),
        pytest.param(False, [1, 2, 1, 2, 1], [44 / 9, 6, 176 / 9, 28.8], [4 / 3, 1.5, 4 / 3, 1.5], id="open-weighted"),
        pytest.param(
            True, [1, 2, 1, 2, 1], [172 / 13, 6, 176 / 9, 15.6], [1.25, 1.5, 4 / 3, 1.5], id="cyclic-weighted"
        ),
    ],
)
def test_smoothing_preserves_bin_boundaries_and_weighted_means(
    cyclic: bool, errors, expected_flux, expected_errors
) -> None:
    data = LCData(
        x_data=[0.0, 0.25, 0.5, 0.75, 1.0],
        y_data=[2.0, 4.0, 8.0, 16.0, 32.0],
        y_err=errors,
        x_unit=u.dimensionless_unscaled,
        y_unit=u.dimensionless_unscaled,
    )
    data.smooth(n_bins=4, radius=1, cyclic_boundaries=cyclic)

    # Exact expectations include points on the right bin boundaries.
    assert_allclose(data.x_data, [0.125, 0.375, 0.625, 0.875])
    assert_allclose(data.y_data, expected_flux)
    if expected_errors is None:
        assert data.y_err is None
    else:
        assert data.y_err is not None
        assert_allclose(data.y_err, expected_errors)


def test_mean_normalization_ignores_unused_fraction() -> None:

    values = {"V": np.array([2.0, 4.0]), "B": np.array([6.0])}
    for kind, scales in [("average", {"V": 3.0, "B": 6.0}), ("global_average", {"V": 4.0, "B": 4.0})]:
        for fraction in (0.0, 1.0, 1.1, np.nan, np.inf, -np.inf):
            shared._validate_lc_normalization(values, kind, fraction)
            normalized, _ = normalize_light_curve(values, kind=kind, top_fraction_to_average=fraction)
            for band, scale in scales.items():
                assert_allclose(normalized[band], values[band] / scale)


def test_extrema_normalization_preserves_selection_and_error_scaling() -> None:

    values = {"V": np.array([2.0, 4.0, 8.0, 16.0]), "B": np.array([3.0, 6.0])}
    errors = {band: np.ones_like(flux) for band, flux in values.items()}
    expected_scales = {
        "maximum": {"V": 28.0 / 3.0, "B": 4.5},
        "global_maximum": {"V": 12.0, "B": 12.0},
        "minimum": {"V": 3.0, "B": 3.0},
    }
    for kind, scales in expected_scales.items():
        with pytest.raises(ValueError, match="fraction"):
            shared._validate_lc_normalization(values, kind, np.nan)
        normalized, normalized_errors = normalize_light_curve(values, errors, kind, 0.5)
        assert normalized_errors is not None
        for band, scale in scales.items():
            assert_allclose(normalized[band], values[band] / scale)
            normalized_error = normalized_errors[band]
            assert normalized_error is not None
            assert_allclose(normalized_error, errors[band] / scale)


def test_extrema_selection_handles_small_and_full_selections() -> None:

    values = {"V": np.array([2.0, 4.0, 8.0])}
    normalized, _ = normalize_light_curve(values, kind="minimum", top_fraction_to_average=0.1)
    assert_allclose(normalized["V"], [1.0, 2.0, 4.0])
    for kind in ("maximum", "global_maximum", "minimum"):
        singleton, _ = normalize_light_curve({"V": np.array([2.0])}, kind=kind)
        assert_allclose(singleton["V"], [1.0])
        # Public callers can request all points; partition indices must remain valid.
        normalized, _ = normalize_light_curve(values, kind=kind, top_fraction_to_average=1.0)
        assert_allclose(normalized["V"], values["V"] / (14.0 / 3.0))


@pytest.mark.parametrize("kind", ["maximum", "global_maximum", "minimum"])
def test_normalization_fraction_boundaries(kind) -> None:

    values = {"V": np.array([2.0, 4.0, 8.0]), "B": np.array([6.0, 12.0])}
    for fraction in (0.0, np.nextafter(1.0, np.inf), 1.1, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="fraction"):
            shared._validate_lc_normalization(values, kind, fraction)
    shared._validate_lc_normalization(values, kind, 1.0)
    normalized, _ = normalize_light_curve(values, kind=kind, top_fraction_to_average=1.0)
    # Global maximum retains floor(total_points / passbands) + 1, even at 1.0.
    scales = {"V": 26.0 / 3.0, "B": 26.0 / 3.0} if kind == "global_maximum" else {"V": 14.0 / 3.0, "B": 9.0}
    for band, scale in scales.items():
        assert_allclose(normalized[band], values[band] / scale)

    fit = least_squares.DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc(errors=[0.1, 0.2, 0.3])})
    observed = fit.y_data[V].copy()
    errors = fit.y_err[V].copy()
    fit.normalize_data(kind=kind, top_fraction_to_average=1.0)
    assert_allclose(fit.y_data[V], observed)
    assert_allclose(fit.y_err[V], errors)
