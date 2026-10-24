"""Shared observation-point predictions for light-curve LS, MCMC and R².

Prepared fitters validate observation coordinates and inverse maps once per
setup; standalone prediction calls validate their inputs on each call. The
pipeline converts coordinates to orbital phases and synthesizes on prepared
phase support. Each passband is then mapped or interpolated onto its own observation
points before mean normalization. This order preserves observation cadence,
duplicates and ordering in the flux arrays supplied to statistical costs.

Structural alignment failures are distinct from physical synthesis failures so
optimizers cannot silently treat malformed data as an improbable binary system.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
from scipy.interpolate import interp1d

from elisa.analytics.models import lc
from elisa.analytics.params import parameters
from elisa.analytics.tools.utils import time_layer_resolver
from elisa.base.error import PredictionAlignmentError
from elisa.observer.utils import normalize_light_curve

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from elisa.observer.observer import Observer

_MIN_CUBIC_SUPPORT = 4


class LCSynthesisError(Exception):
    """Wrap a ValueError, RuntimeError or TypeError from physical synthesis.

    The original exception is retained as ``__cause__``. LS applies its existing
    synthesis penalty; MCMC and R² restore the original exception for their
    existing handling. Alignment errors outside synthesis are not wrapped.
    """


def _fail(message: str) -> NoReturn:
    """Raise a structural prediction error with the supplied diagnostic.

    :param message: Description of the incompatible data or sampling support.
    :raises PredictionAlignmentError: Always; this function never returns.
    """
    raise PredictionAlignmentError(message)


def _validate_band(
    band: str, xs: NDArray, observed: Any, reducer: Any,
    x_data: dict | None, y_err: dict | None,
) -> NDArray:
    """Validate one passband's inverse coordinate map and uncertainty shape.

    A reducer indexes the original coordinate union, before phase folding. Its
    entries must reproduce the band's observation order, including duplicates.
    A scalar uncertainty is accepted for the existing missing-error fallback;
    an uncertainty array must have exactly one entry per observed flux.

    :param band: Passband key used in the observation dictionaries.
    :param xs: One-dimensional union of original phase or JD coordinates.
    :param observed: Nonempty, one-dimensional observed flux array.
    :param reducer: Integer indices mapping this band's observations into xs.
    :param x_data: Original coordinates by band, or None to omit the exact
        coordinate correspondence check.
    :param y_err: Uncertainties by band, or None to omit their shape check.
    :return: Validated reducer as a NumPy array, without reordering its entries.
    :raises PredictionAlignmentError: If lengths, indices, coordinate order or
        uncertainty shape are incompatible.
    """
    observed, reducer = np.asarray(observed), np.asarray(reducer)
    if observed.ndim != 1 or observed.size == 0 or reducer.shape != observed.shape:
        _fail(f"LC observation/reducer lengths differ for {band}")
    if reducer.dtype.kind not in "iu":
        _fail(f"Invalid LC observation reducer for {band}")
    if np.any((reducer < 0) | (reducer >= xs.size)):
        _fail(f"Invalid LC observation reducer for {band}")
    if x_data is not None and not np.array_equal(xs[reducer], x_data[band]):
        _fail(f"LC observation coordinates do not match reducer for {band}")
    # Missing-error estimation supplies a scalar shared by the band's data.
    if y_err is not None and np.shape(y_err[band]) not in ((), observed.shape):
        _fail(f"LC uncertainty length differs from observations for {band}")
    return reducer


def validate_observations(
    xs: NDArray, y_data: dict, x_data_reducer: dict,
    x_data: dict | None, y_err: dict | None,
) -> dict:
    """Validate shared observation coordinates and all per-passband reducers.

    :param xs: Nonempty, finite vector of original phase or JD coordinates.
    :param y_data: Observed flux arrays keyed by passband.
    :param x_data_reducer: Inverse coordinate maps keyed by the same passbands.
    :param x_data: Optional original coordinates for exact reducer validation.
    :param y_err: Optional uncertainties for per-band shape validation.
    :return: Dictionary of validated NumPy reducer arrays.
    :raises PredictionAlignmentError: If the coordinate vector, passband keys
        or any band's observation/reducer alignment is invalid.
    """
    if xs.ndim != 1 or xs.size == 0 or not np.all(np.isfinite(xs)):
        _fail("LC observation coordinates must be a nonempty finite vector")
    bands = set(y_data)
    if not bands or set(x_data_reducer) != bands:
        _fail("LC observation and reducer passbands differ")
    if x_data is not None and set(x_data) != bands:
        _fail("LC coordinate and flux passbands differ")
    if y_err is not None and set(y_err) != bands:
        _fail("LC uncertainty and flux passbands differ")
    return {
        band: _validate_band(band, xs, y_data[band], x_data_reducer[band], x_data, y_err)
        for band in y_data
    }


def _synthesis_support(phases: NDArray, interp_treshold: int, fit_xs: NDArray | None) -> NDArray:
    """Select prepared support or the common direct/uniform fallback grid.

    Prepared uniform, adaptive or manual support is already in orbital phases
    and is retained unchanged. Without prepared support, observations below the
    threshold are synthesized directly. At or above the threshold, use T+2
    uniform points on [-1/T, 1+1/T], matching fitting setup. Interpolation-specific
    checks for distinct phases and observation coverage occur in the caller.

    :param phases: Observation-coordinate union converted to orbital phases.
    :param interp_treshold: Positive sampling threshold T from fitting setup.
    :param fit_xs: Prepared phase support, or None to select the fallback grid.
    :return: Nonempty, finite, one-dimensional synthesis support array.
    :raises PredictionAlignmentError: If the selected support is malformed.
    """
    if fit_xs is None:
        if phases.size >= interp_treshold:
            diff = 1.0 / interp_treshold
            fit_xs = np.linspace(-diff, 1.0 + diff, num=interp_treshold + 2)
        else:
            fit_xs = phases
    support = np.asarray(fit_xs)
    if support.ndim != 1 or support.size == 0 or not np.all(np.isfinite(support)):
        _fail("LC synthesis support must be a nonempty finite vector")
    return support


def predict_light_curves(
    x_data_reduced: NDArray, y_data: dict, x_data_reducer: dict,
    discretization: float, observer: Observer, interp_treshold: int,
    model_parameters: dict, *, fit_xs: NDArray | None = None,
    x_data: dict | None = None, y_err: dict | None = None,
    atmosphere_model: dict | None = None, limb_darkening_coefficients: dict | None = None,
    synthesis: Callable[..., dict] = lc.synthetic_binary,
    _validated_reducers: dict | None = None,
) -> dict[str, NDArray]:
    """Synthesize, map to each band's observations, then normalize by its mean.

    Reducers refer to the original (possibly JD) coordinates, before phase
    folding. Prepared support is already in phases. Exact coordinate equality,
    including order, is required to use the direct branch. Otherwise, cubic
    interpolation evaluates each curve at that band's reduced observation
    phases; equal array lengths alone never select the direct branch.

    Mean normalization occurs only after mapping to observation points. Thus
    repeated observations retain their multiplicity in the normalization, and
    each returned array has exactly the shape and order of its observed fluxes.
    Observed fluxes and uncertainties are used for alignment validation here;
    cost evaluation and observation normalization remain the caller's work.

    :param x_data_reduced: Union of original observation coordinates, in phases
        or JD as determined by the supplied ephemeris parameters.
    :param y_data: Observed flux arrays keyed by passband.
    :param x_data_reducer: Per-band inverse maps into x_data_reduced.
    :param discretization: Stellar surface discretization factor passed to
        synthesis, using the synthesis engine's degree convention.
    :param observer: Observer configured for the observation passbands and
        system class; synthesis updates its binary system.
    :param interp_treshold: Positive support-selection threshold T.
    :param model_parameters: Flat physical parameter dictionary with evaluated
        constraints and any ephemeris needed for JD-to-phase conversion.
    :param fit_xs: Optional prepared uniform, adaptive or manual phase support.
    :param x_data: Optional original coordinates by band for exact validation
        of the inverse maps before phase conversion.
    :param y_err: Optional per-band uncertainty arrays or scalar fallbacks.
    :param atmosphere_model: Optional per-component atmosphere overrides.
    :param limb_darkening_coefficients: Optional per-component, per-passband
        custom limb-darkening coefficients.
    :param synthesis: Physical synthesis callable accepting support phases,
        discretization, observer and flat model keyword arguments. Defaults to
        the real binary light-curve synthesis engine.
    :param _validated_reducers: Internal per-setup validated mappings. Prepared
        fitters supply this to avoid rescanning static observations. Standalone
        calls omit it and validate their inputs on every call. Manual changes to
        prepared observations require fitter.validate_observations() or set_up().
    :return: Per-band mean-normalized predictions at the observation points.
    :raises PredictionAlignmentError: If observations, reducers, interpolation
        support or returned synthesis arrays do not align. Callers must let
        this structural error escape physical-state penalty handlers.
    :raises LCSynthesisError: If synthesis raises ValueError, RuntimeError or
        TypeError; the original exception is retained as __cause__. Other
        synthesis exceptions propagate unchanged.
    """
    xs = np.asarray(x_data_reduced)
    reducers = (
        validate_observations(xs, y_data, x_data_reducer, x_data, y_err)
        if _validated_reducers is None else _validated_reducers
    )
    phases, kwargs = time_layer_resolver(xs, pop=False, **model_parameters)
    support = _synthesis_support(phases, interp_treshold, fit_xs)
    interpolate = not np.array_equal(support, phases)
    if interpolate:
        if support.size < _MIN_CUBIC_SUPPORT or np.unique(support).size != support.size:
            _fail("Cubic LC interpolation needs at least four distinct support phases")
        if np.min(phases) < np.min(support) or np.max(phases) > np.max(support):
            _fail("LC synthesis support does not cover observation phases")

    kwargs = parameters.extend_json_with_atm_params(
        kwargs, atmosphere_model=atmosphere_model,
        limb_darkening_coefficients=limb_darkening_coefficients,
    )
    try:
        raw = synthesis(support, discretization, observer, **kwargs)
    except PredictionAlignmentError:
        raise
    except (ValueError, RuntimeError, TypeError) as exc:
        raise LCSynthesisError(str(exc)) from exc
    if raw.keys() != y_data.keys():
        _fail("LC synthesis and observation passbands differ")
    predictions = {}
    for band, raw_curve in raw.items():
        curve = np.asarray(raw_curve)
        if curve.shape != support.shape:
            _fail(f"LC synthetic length differs from support for {band}")
        predictions[band] = (
            interp1d(support, curve, kind="cubic")(phases[reducers[band]])
            if interpolate else curve[reducers[band]]
        )
        if predictions[band].shape != np.shape(y_data[band]):
            _fail(f"LC prediction length differs from observations for {band}")
    predictions, _ = normalize_light_curve(predictions, kind="average")
    return predictions
