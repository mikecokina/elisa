from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from elisa.utils import is_empty

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray

    from elisa.types import Float


def normalize_light_curve(
        y_data: Mapping[str, ArrayLike],
        y_err: Mapping[str, ArrayLike | None] | None = None,
        kind: str = "global_maximum",
        top_fraction_to_average: Float = 0.1,
) -> tuple[dict[str, NDArray[Float]], dict[str, NDArray[Float] | None] | None]:
    """Normalize light curves using the selected normalization strategy.

    Supported normalization kinds are:

    - ``"average"`` - each curve is normalized by its own mean value
    - ``"global_average"`` - all curves are normalized by a shared global mean
    - ``"maximum"`` - each curve is normalized by the average of its top fraction
    - ``"global_maximum"`` - all curves are normalized by the global top-fraction average
    - ``"minimum"`` - each curve is normalized by the average of its bottom fraction

    :param y_data: Dictionary of curves in the form ``{filter_name: values}``.
    :type y_data: Mapping[str, ArrayLike]
    :param y_err: Optional dictionary of curve uncertainties in the form
        ``{filter_name: errors}``. Defaults to None.
    :type y_err: Mapping[str, ArrayLike | None] | None
    :param kind: Normalization kind. Must be one of the supported kinds listed above.
        Defaults to "global_maximum".
    :type kind: str
    :param top_fraction_to_average: Fraction of points used when computing top-fraction
        or bottom-fraction averages. Expected to be in the interval ``(0, 1)``.
        Defaults to 0.1.
    :type top_fraction_to_average: Float
    :returns: Tuple containing normalized curves and normalized errors. The first element
        is a dictionary of normalized flux curves (dict[str, NDArray[Float]]), and the
        second element is a dictionary of normalized error curves or None if no errors
        were provided (dict[str, NDArray[Float] | None] | None).
    :rtype: tuple[dict[str, NDArray[Float]], dict[str, NDArray[Float] | None] | None]
    :raises ValueError: If ``kind`` is not one of the supported normalization modes.
    """
    valid_arguments = ["average", "global_average", "maximum", "global_maximum", "minimum"]

    y_data_arrays: dict[str, NDArray[Float]] = {
        key: np.asarray(val, dtype=float) for key, val in y_data.items()
    }

    if kind == "average":
        coeff = {key: np.mean(val) for key, val in y_data_arrays.items()}
    elif kind == "global_average":
        c = np.mean(np.concatenate(list(y_data_arrays.values())))
        coeff = dict.fromkeys(y_data_arrays, c)
    elif kind == "maximum":
        n = {key: int(top_fraction_to_average * len(val)) + 1 for key, val in y_data_arrays.items()}
        coeff = {
            key: np.average(val[np.argsort(val)[-n[key]:]])
            for key, val in y_data_arrays.items()
        }
    elif kind == "global_maximum":
        vals = np.concatenate(list(y_data_arrays.values()))
        n = int(top_fraction_to_average * len(vals) / len(y_data_arrays)) + 1
        c = np.average(vals[np.argsort(vals)[-n:]])
        coeff = dict.fromkeys(y_data_arrays, c)
    elif kind == "minimum":
        n = {key: int(top_fraction_to_average * len(val)) for key, val in y_data_arrays.items()}
        coeff = {
            key: np.average(val[np.argsort(val)[:n[key]]])
            for key, val in y_data_arrays.items()
        }
    else:
        msg = f"Argument `kind` = {kind} is not one of the valid arguments {valid_arguments}"
        raise ValueError(msg)

    normalized_data = {key: val / coeff[key] for key, val in y_data_arrays.items()}

    normalized_err: dict[str, NDArray[Float] | None] | None
    if is_empty(y_err):
        normalized_err = None
    else:
        normalized_err = {
            key: np.asarray(val, dtype=float) / coeff[key] if not is_empty(val) else None
            for key, val in y_err.items()
        }

    return normalized_data, normalized_err


def adjust_flux_for_distance(
        curves: Mapping[str, ArrayLike],
        distance: Float,
) -> dict[str, NDArray[Float]]:
    """Scale flux curves to the specified observer distance.

    Scales flux curves by the inverse square of the provided distance. This function
    is typically used to correct for distance-dependent flux attenuation in astronomical
    observations.

    :param curves: Band-wise flux curves in the form ``{band_name: flux_values}``.
    :type curves: Mapping[str, ArrayLike]
    :param distance: Distance to the observer.
    :type distance: Float
    :returns: Distance-corrected band-wise flux curves with the same structure as input
        (dict[str, NDArray[Float]]).
    :rtype: dict[str, NDArray[Float]]
    """
    d_squared = np.power(distance, 2)
    return {
        band: np.asarray(curve, dtype=float) / d_squared
        for band, curve in curves.items()
    }


def convert_to_magnitudes(
        curves: Mapping[str, ArrayLike],
        zero_points: Mapping[str, Mapping[str, Float | None]],
) -> dict[str, NDArray[Float]]:
    """Convert flux curves to magnitudes.

    Converts flux curves to magnitude space using the provided zero-point calibration
    data. The conversion uses the standard magnitude formula: m = m_ref - 2.5 * log10(f/f_ref).

    :param curves: Band-wise flux curves in the form ``{band_name: flux_values}``.
    :type curves: Mapping[str, ArrayLike]
    :param zero_points: Calibration data containing ``reference_magnitudes`` and ``fluxes``
        for each band, in the form ``{calibration_key: {band_name: value}}``.
    :type zero_points: Mapping[str, Mapping[str, Float | None]]
    :returns: Band-wise magnitude curves with the same structure as input
        (dict[str, NDArray[Float]]).
    :rtype: dict[str, NDArray[Float]]
    :raises ValueError: If a reference magnitude is not available for a requested band.
    """
    ret_dict: dict[str, NDArray[Float]] = {}

    for band, curve in curves.items():
        reference_magnitude = zero_points["reference_magnitudes"][band]
        if reference_magnitude is None:
            msg = f"Calibration reference magnitude is not available for the filter {band}"
            raise ValueError(msg)

        flux_curve = np.asarray(curve, dtype=float)
        ret_dict[band] = reference_magnitude - 2.5 * np.log10(
            flux_curve / zero_points["fluxes"][band],
        )

    return ret_dict
