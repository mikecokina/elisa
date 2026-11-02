"""Observation uncertainties, validation and fitter normalization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import units as u
from elisa.analytics import LCData, RVData
from elisa.analytics.binary_fit import least_squares, mcmc, shared
from elisa.analytics.dataset.utils import convert_flux_error
from elisa.analytics.params.parameters import BinaryInitialParameters
# noinspection PyProtectedMember
from elisa.analytics.tools.utils import _radialcurves_mean_error
from elisa.base.error import ElisaAnalyticsDataError, ElisaAnalyticsTaskError
from elisa.binary_system.curves.community import RadialVelocitySystem
from tests.tests_analytics.fitting_helpers import (
    LC_PHASES,
    V,
    lc_setup,
    make_lc,
    normalized,
    prepare_lc_fits,
    rv_problem,
)

LC_CLASSES = (least_squares.DetachedLightCurveFit, mcmc.DetachedLightCurveFit)
RV_CLASSES = (least_squares.CentralRadialVelocity, mcmc.CentralRadialVelocity)


@pytest.mark.parametrize(("unit", "factor"), [(u.mag, 1.0), (u.mmag, 1000.0)])
def test_pointwise_magnitude_uncertainties(unit, factor) -> None:
    magnitudes = np.array([9.0, 10.0, 11.0])
    sigma_mag = np.array([0.01, 0.02, 0.03])
    data = make_lc(factor * magnitudes, factor * sigma_mag, y_unit=unit, reference_magnitude=10.0)
    expected_flux = 10.0 ** ((10.0 - magnitudes) / 2.5)
    assert_allclose(data.y_data, expected_flux, rtol=1e-14)
    assert data.y_err is not None
    assert_allclose(data.y_err, np.log(10.0) / 2.5 * expected_flux * sigma_mag, rtol=1e-14)
    missing = make_lc(factor * magnitudes, y_unit=unit, reference_magnitude=10.0)
    assert missing.y_err is None
    for cls in LC_CLASSES:
        fit = cls()
        lc_setup(fit, {V: missing})
        assert_allclose(fit.y_err[V], 0.05)


def test_magnitude_zero_point_alone_cannot_determine_flux_errors() -> None:
    with pytest.raises(ElisaAnalyticsDataError, match="pointwise"):
        convert_flux_error([0.01], u.mag, zero_point=10.0)


@pytest.mark.parametrize("cls", LC_CLASSES)
@pytest.mark.parametrize("errors", [None, [0.02, 0.04, 0.03]])
def test_lc_normalization_and_repeated_setup(cls, errors) -> None:
    data = make_lc(errors=errors)
    original = data.y_data.copy()
    fit = cls()
    with patch.object(shared.logger, "warning") as warning:
        lc_setup(fit, {V: data})
    assert warning.call_count == (1 if errors is None else 0)
    if errors is None:
        message = warning.call_args.args[0]
        assert "heuristic weights" in message
        assert "credible intervals depend" in message
    assert_allclose(fit.y_data[V], original / 3.0)
    assert_allclose(fit.y_err[V], 0.05 if errors is None else np.asarray(errors) / 3.0)
    assert_array_equal(data.y_data, original)
    if errors is None:
        assert data.y_err is None
    else:
        assert data.y_err is not None
        assert_array_equal(data.y_err, errors)
    replacement = make_lc([8.0, 12.0], [0.2, 0.4])
    lc_setup(fit, {V: replacement})
    assert_allclose(fit.y_err[V], [0.02, 0.04])
    assert fit.num_of_points == {V: 2}
    assert fit.fit_xs is None
    assert len(fit.x_data_reducer[V]) == 2


@pytest.mark.parametrize("cls", RV_CLASSES)
@pytest.mark.parametrize("supplied", [False, True])
def test_rv_component_scales_and_repeated_setup(cls, supplied: bool) -> None:
    cfg, _ = rv_problem()
    values = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    data = {
        component: RVData(
            x_data=np.arange(5) / 5.0,
            y_data=velocities,
            y_err=np.full(5, 0.3) if supplied else None,
            x_unit=u.dimensionless_unscaled,
            y_unit=u.km / u.s,
        )
        for component, velocities in [("primary", values), ("secondary", 12000.0 + 3.0 * values)]
    }
    fit = cls()
    for component in ("secondary", "primary"):
        subset = {component: data[component]}
        fit.set_up(BinaryInitialParameters(**cfg), subset, observer_system_cls=RadialVelocitySystem)
        assert set(fit.y_err) == {component}
        expected = (
            np.full(5, 300.0)
            if supplied
            else np.full(5, 0.05 * 1.482602218505602 * 2000.0 * (3 if component == "secondary" else 1))
        )
        assert_allclose(fit.y_err[component], expected)
    fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    if not supplied:
        assert_allclose(fit.y_err["secondary"], 3.0 * fit.y_err["primary"])


@pytest.mark.parametrize("values", [[0.0, 0.0], [12000.0] * 3, [1.0], [], [0.0, np.nan], [0.0, np.inf]])
def test_uninformative_rv_requires_errors(values) -> None:
    with pytest.raises(ElisaAnalyticsDataError, match="supply explicit positive y_err"):
        _radialcurves_mean_error(values)


@pytest.mark.parametrize("amplitude", [1e-200, 1e-10, 1e200])
def test_rv_scales_small_and_large_variations(amplitude: float) -> None:
    assert_allclose(
        _radialcurves_mean_error(amplitude * np.array([-1.0, 0.0, 1.0])),
        0.05 * 1.482602218505602 * amplitude,
        atol=0.0,
        rtol=2e-14,
    )


def test_rv_translation_mad_outliers_and_tied_fallback() -> None:
    values = np.array([-300.0, -100.0, 0.0, 100.0, 300.0, 10000.0])
    expected = 0.05 * 1.482602218505602 * np.median(np.abs(values - np.median(values)))
    for offset in (0.0, -1000.0, 1e9):
        assert_allclose(_radialcurves_mean_error(values + offset), expected, rtol=2e-14)
    tied = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    assert_allclose(_radialcurves_mean_error(tied), 0.05 * np.std(tied, ddof=1))
    near_constant = 12000.0 + np.arange(5) * 2.0**-30
    assert _radialcurves_mean_error(near_constant) > 0
    assert_allclose(_radialcurves_mean_error(near_constant), _radialcurves_mean_error(near_constant - 12000.0))
    core = [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert_allclose(_radialcurves_mean_error([*core, 100.0]), _radialcurves_mean_error([*core, 1e6]))


@pytest.mark.parametrize("cls", RV_CLASSES)
def test_constant_rv_with_explicit_errors(cls) -> None:
    cfg, data = rv_problem()
    data["primary"].y_data[:] = 12000.0
    fit = cls()
    fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    assert_allclose(fit.y_err["primary"], 100.0)
    data["primary"].y_err = None
    with pytest.raises(ElisaAnalyticsDataError, match="supply explicit positive y_err"):
        fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)


@pytest.mark.parametrize(("dataset_cls", "unit"), [(LCData, u.dimensionless_unscaled), (RVData, u.m / u.s)])
@pytest.mark.parametrize("errors", [[], [0.0, 0.1], [-0.1, 0.1], [np.nan, 0.1], [np.inf, 0.1], [0.1], [[0.1, 0.1]]])
def test_invalid_supplied_errors(dataset_cls, unit, errors) -> None:
    with pytest.raises(ElisaAnalyticsDataError):
        dataset_cls(x_data=[0.0, 0.5], y_data=[1.0, 2.0], y_err=errors, x_unit=u.dimensionless_unscaled, y_unit=unit)


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [[1.0, 2.0]], 1.0])
def test_invalid_observation_arrays(values) -> None:
    with pytest.raises(ElisaAnalyticsDataError):
        LCData(x_data=values, y_data=values, x_unit=u.dimensionless_unscaled, y_unit=u.dimensionless_unscaled)


@pytest.mark.parametrize("cls", [*LC_CLASSES, *RV_CLASSES])
def test_setup_rejects_mutated_dataset(cls) -> None:
    if cls in LC_CLASSES:
        data = make_lc(errors=[0.1] * 3)
        assert data.y_err is not None
        data.y_err[0] = 0.0
        with pytest.raises(ElisaAnalyticsDataError):
            lc_setup(cls(), {V: data})
    else:
        cfg, data = rv_problem()
        assert data["primary"].y_err is not None
        data["primary"].y_err[0] = 0.0
        with pytest.raises(ElisaAnalyticsDataError):
            cls().set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)


@pytest.mark.parametrize("values", [[-1.0, 1.0], [-2.0, -1.0]])
def test_lc_invalid_normalization(values) -> None:
    with pytest.raises(ElisaAnalyticsDataError):
        lc_setup(least_squares.DetachedLightCurveFit(), {V: make_lc(values, [0.1, 0.1])})


def test_file_validation_precedes_downselection_and_smoothing_preserves_none(tmp_path: Path) -> None:
    path = tmp_path / "observations.dat"
    path.write_text("0 1 .1\n.1 1 0\n.2 1 .1\n")
    with pytest.raises(ElisaAnalyticsDataError, match="strictly positive"):
        LCData.load_from_file(str(path), downselect_ratio=0.5)
    for bad in ("nan", "inf", "bad"):
        path.write_text(f"0 1\n.2 {bad}\n")
        with pytest.raises(ElisaAnalyticsDataError):
            LCData.load_from_file(str(path), data_columns=(0, 1))
    path.write_text("0 1\n.2 2\n.4 3\n")
    data = LCData.load_from_file(str(path), data_columns=(0, 1))
    data.smooth(n_bins=2, radius=1)
    assert data.y_err is None


@pytest.mark.parametrize("base", [shared.AbstractLCFit, shared.AbstractRVFit])
def test_slotted_subclasses_use_declared_attributes(base) -> None:
    class Fit(base):
        __slots__ = ()

        def fit(self, *args, **kwargs):
            raise NotImplementedError

    fit = Fit()
    assert not hasattr(fit, "__dict__")
    fit.atmosphere_model = {"primary": "bb"}
    fit.limb_darkening_coefficients = None
    if base is shared.AbstractLCFit:
        lc_setup(fit, {V: make_lc()})
        assert fit._validated_reducers is not None
    else:
        cfg, data = rv_problem()
        fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    slots = [name for cls in type(fit).__mro__ for name in cls.__dict__.get("__slots__", ())]
    assert len(slots) == len(set(slots))
    with pytest.raises(AttributeError):
        fit.undeclared = 1


@pytest.mark.parametrize("cls", RV_CLASSES)
@pytest.mark.parametrize("supplied", [False, True])
def test_actual_rv_evaluations_do_not_validate_inputs_again(cls, supplied: bool) -> None:
    cfg, data = rv_problem()
    if not supplied:
        for dataset in data.values():
            dataset.y_err = None
    fit = cls()
    fit.set_up(BinaryInitialParameters(**cfg), data, observer_system_cls=RadialVelocitySystem)
    with patch.object(shared.DataSet, "check_data_validity", side_effect=AssertionError("Repeated validation")):
        for _ in range(2):
            result = (
                fit.likelihood(normalized(fit))
                if cls is mcmc.CentralRadialVelocity
                else fit.central_rv_model_to_fit(normalized(fit))
            )
            assert np.all(np.isfinite(result))


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.requires_ld(passbands=(V, "bolometric"), laws=("linear",))
def test_real_lc_evaluations_with_optional_errors(science_settings: None, supplied: bool) -> None:
    fits = prepare_lc_fits({V: LC_PHASES}, missing_errors=not supplied)
    assert_allclose(fits[0].y_data[V], fits[1].y_data[V])
    assert_allclose(fits[0].y_err[V], fits[1].y_err[V])
    with (
        patch.object(shared.DataSet, "check_data_validity", side_effect=AssertionError("Repeated validation")),
        patch.object(shared, "validate_observations", side_effect=AssertionError("Repeated mapping validation")),
    ):
        for _ in range(2):
            assert np.all(np.isfinite(fits[0].model_to_fit(normalized(fits[0]))))
            assert np.isfinite(fits[1].likelihood(normalized(fits[1])))


@pytest.mark.parametrize("cls", RV_CLASSES)
def test_empty_and_invalid_component_datasets(cls) -> None:
    cfg, data = rv_problem()
    for bad, error_type in (({}, ElisaAnalyticsDataError), ({"tertiary": data["primary"]}, ElisaAnalyticsTaskError)):
        with pytest.raises(error_type):
            cls().set_up(BinaryInitialParameters(**cfg), bad, observer_system_cls=RadialVelocitySystem)


def test_normalization_fraction_validated_during_setup_operations() -> None:
    fit = least_squares.DetachedLightCurveFit()
    lc_setup(fit, {V: make_lc()})
    for fraction in (0.0, 1.1, -0.1, np.nan, np.inf, -np.inf):
        with pytest.raises(ElisaAnalyticsTaskError, match="fraction"):
            fit.normalize_data(top_fraction_to_average=fraction)
