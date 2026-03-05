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
    # noinspection GrazieInspection
    """Normalize light curves using the selected normalization strategy.

    Supported normalization kinds are:

    - ``"average"`` - each curve is normalized by its own mean value
    - ``"global_average"`` - all curves are normalized by a shared global mean
    - ``"maximum"`` - each curve is normalized by the average of its top fraction
    - ``"global_maximum"`` - all curves are normalized by the global top-fraction average
    - ``"minimum"`` - each curve is normalized by the average of its bottom fraction

    :param y_data: Mapping[str, ArrayLike]
        Dictionary of curves in the form ``{filter_name: values}``.
    :param y_err: Mapping[str, ArrayLike | None] | None
        Optional dictionary of curve uncertainties in the form
        ``{filter_name: errors}``.
    :param kind: str
        Normalization kind.
    :param top_fraction_to_average: Float
        Fraction of points used when computing top-fraction or bottom-fraction
        averages. Expected to be in the interval ``(0, 1)``.
    :returns: tuple[dict[str, NDArray[Float]], dict[str, NDArray[Float] | None] | None]
        Tuple containing normalized curves and normalized errors.
    :raises ValueError:
        If ``kind`` is not one of the supported normalization modes.
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

    :param curves: Mapping[str, ArrayLike]
        Band-wise flux curves.
    :param distance: Float
        Distance to the observer.
    :returns: dict[str, NDArray[Float]]
        Distance-corrected band-wise flux curves.
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

    :param curves: Mapping[str, ArrayLike]
        Band-wise flux curves.
    :param zero_points: Mapping[str, Mapping[str, Float | None]]
        Calibration data containing ``reference_magnitudes`` and ``fluxes``.
    :returns: dict[str, NDArray[Float]]
        Band-wise magnitude curves.
    :raises ValueError:
        If a reference magnitude is not available for a requested band.
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
