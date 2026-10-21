"""Shared helpers for loading LC fit-result JSON into the modelling and visualization UI forms.

Contains both the pure-Python parsing / conversion logic and the Gradio handler factory
so both the LC Modelling and the System Visualization tabs can reuse identical
functionality without duplication.

Public surface:

- :func:`load_model_params_from_json` - parse JSON, return three param dicts
- :func:`make_json_load_handler` - factory that returns a Gradio upload handler
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from elisa import const

if TYPE_CHECKING:
    from collections.abc import Callable

    from elisa.types import Float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _masses_from_community(
    semi_major_axis_solrad: Float,
    mass_ratio: Float,
    period_days: Float,
) -> tuple[Float, Float]:
    r"""Derive individual component masses from community fit parameters.

    Applies Kepler's third law in SI units and returns masses in solar masses:

    .. math::

        M_1 + M_2 = \frac{4\pi^2 a^3}{G P^2}, \quad
        M_1 = \frac{M_1+M_2}{1+q}, \quad M_2 = q M_1

    :param semi_major_axis_solrad: Semi-major axis in solar radii
        (``const.SOLAR_RADIUS`` used for conversion).
    :type semi_major_axis_solrad: Float
    :param mass_ratio: Mass ratio :math:`q = M_2 / M_1`.
    :type mass_ratio: Float
    :param period_days: Orbital period in days.
    :type period_days: Float
    :returns: Tuple ``(M1, M2)`` in solar masses.
    :rtype: tuple[Float, Float]
    """
    a_m = semi_major_axis_solrad * const.SOLAR_RADIUS
    p_s = period_days * 86_400.0  # 1 day = 86 400 s (exact SI definition)
    m_total_kg = 4.0 * math.pi**2 * a_m**3 / (const.G * p_s**2)
    m_total = m_total_kg / const.SOLAR_MASS
    m1 = m_total / (1.0 + mass_ratio)
    m2 = mass_ratio * m1
    return m1, m2


def _get_fit_val(section: dict[str, object], name: str) -> Float | None:
    """Extract the ``"value"`` field from one fit-result parameter entry.

    :param section: One section of the fit-result JSON
        (e.g. ``data["primary"]``).
    :type section: dict[str, object]
    :param name: Parameter name key.
    :type name: str
    :returns: Numeric value, or ``None`` if absent.
    :rtype: Float | None
    """
    entry = section.get(name)
    if isinstance(entry, dict):
        raw = entry.get("value")
        return float(raw) if raw is not None else None
    return None


def _extract_comp_params(section: dict[str, object]) -> dict[str, object]:
    """Extract modelling-relevant component parameters from one fit-result section.

    Reads ``t_eff``, ``surface_potential``, ``synchronicity``,
    ``gravity_darkening``, ``albedo``, and ``metallicity``.  Keys whose
    value is absent in the JSON are omitted so the form keeps its default.

    :param section: One component section of the fit-result JSON
        (e.g. ``data["primary"]``).
    :type section: dict[str, object]
    :returns: Flat dict suitable for a star-inputs form.
    :rtype: dict[str, object]
    """
    names = (
        "t_eff",
        "surface_potential",
        "synchronicity",
        "gravity_darkening",
        "albedo",
        "metallicity",
    )
    return {name: v for name in names if (v := _get_fit_val(section, name)) is not None}


def _extract_system_params(section: dict[str, object]) -> dict[str, object]:
    """Extract modelling-relevant system parameters from the fit-result system section.

    Reads ``inclination``, ``period``, ``eccentricity``,
    ``argument_of_periastron``, ``phase_shift``, ``additional_light``,
    and ``primary_minimum_time``.  ``gamma`` and ``distance`` are not
    present in fit results and are therefore not populated.

    :param section: The ``system`` section of the fit-result JSON.
    :type section: dict[str, object]
    :returns: Flat dict suitable for a system-inputs form.
    :rtype: dict[str, object]
    """
    names = (
        "inclination",
        "period",
        "eccentricity",
        "argument_of_periastron",
        "phase_shift",
        "additional_light",
        "primary_minimum_time",
    )
    return {name: v for name in names if (v := _get_fit_val(section, name)) is not None}


# ---------------------------------------------------------------------------
# Public API - parsing
# ---------------------------------------------------------------------------


def load_model_params_from_json(
    path: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Load modelling form parameters from a saved LC fit-result JSON.

    Supports both fitting approaches:

    - **Standard** - individual component masses are read directly from the
      ``primary`` and ``secondary`` sections.
    - **Community** - the semi-major axis, mass ratio, and period from the
      ``system`` section are converted to individual masses via Kepler's
      third law before populating the form.

    System parameters that have no equivalent in the modelling form
    (e.g. ``gamma``, ``distance``) are left out so the form keeps its
    existing defaults.

    :param path: Absolute path to the JSON file produced by
        :meth:`~elisa.analytics.LCBinaryAnalyticsTask.save_result`.
    :type path: str
    :returns: Tuple ``(primary_params, secondary_params, system_params)``
        where each dict is keyed by the field names used by
        :mod:`~elisa.ui.components.star_inputs` and
        :mod:`~elisa.ui.components.system_inputs` respectively.
    :rtype: tuple[dict[str, object], dict[str, object], dict[str, object]]
    :raises ValueError: If the file cannot be read or lacks a ``"system"`` key.
    """
    try:
        data: dict[str, object] = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        msg = f"Cannot read result file: {exc}"
        raise ValueError(msg) from exc

    if "system" not in data:
        msg = "File does not look like an ELISa LC result - missing 'system' key."
        raise ValueError(msg)

    system_data: dict[str, object] = data.get("system") or {}  # type: ignore[assignment]
    primary_data: dict[str, object] = data.get("primary") or {}  # type: ignore[assignment]
    secondary_data: dict[str, object] = data.get("secondary") or {}  # type: ignore[assignment]

    primary_params = _extract_comp_params(primary_data)
    secondary_params = _extract_comp_params(secondary_data)

    is_community = "mass_ratio" in system_data
    if is_community:
        a = _get_fit_val(system_data, "semi_major_axis")
        q = _get_fit_val(system_data, "mass_ratio")
        period = _get_fit_val(system_data, "period")
        if a is not None and q is not None and period is not None:
            m1, m2 = _masses_from_community(a, q, period)
            primary_params["mass"] = m1
            secondary_params["mass"] = m2
    else:
        if (m1 := _get_fit_val(primary_data, "mass")) is not None:
            primary_params["mass"] = m1
        if (m2 := _get_fit_val(secondary_data, "mass")) is not None:
            secondary_params["mass"] = m2

    return primary_params, secondary_params, _extract_system_params(system_data)


# ---------------------------------------------------------------------------
# Public API - Gradio handler factory
# ---------------------------------------------------------------------------


def make_json_load_handler(
    prim_keys: tuple[str, ...],
    sec_keys: tuple[str, ...],
    sys_keys: tuple[str, ...],
) -> Callable[..., list[object]]:
    """Return a Gradio upload handler that populates a modelling form from a fit JSON.

    The returned callable accepts a single Gradio file object and emits one
    ``gr.update`` per output component in
    ``prim_comps + sec_comps + sys_comps`` order (matching *prim_keys*,
    *sec_keys*, *sys_keys*).  Fields absent from the JSON get an empty
    ``gr.update()`` so the form keeps its current value.

    Both **Standard** and **Community** fit JSONs are accepted - community
    parameters are automatically converted to individual masses via Kepler's
    third law (see :func:`load_model_params_from_json`).

    :param prim_keys: Ordered field names for the primary star form
        (see :data:`~elisa.ui.components.star_inputs.FIELD_ORDER`).
    :type prim_keys: tuple[str, ...]
    :param sec_keys: Ordered field names for the secondary star form.
    :type sec_keys: tuple[str, ...]
    :param sys_keys: Ordered field names for the binary-system form
        (see :data:`~elisa.ui.components.system_inputs.FIELD_ORDER`).
    :type sys_keys: tuple[str, ...]
    :returns: A callable suitable for ``gr.File.upload(fn=...)``.
    :rtype: Callable[..., list[object]]
    """

    def handler(json_file: object) -> list[object]:
        path: str | None = getattr(json_file, "name", None)
        if path is None:
            msg = "No file received."
            raise gr.Error(msg)

        try:
            prim, sec, sys_ = load_model_params_from_json(path)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        updates: list[object] = []
        for keys, params in ((prim_keys, prim), (sec_keys, sec), (sys_keys, sys_)):
            updates.extend(
                gr.update(value=params[key]) if key in params else gr.update()
                for key in keys
            )
        return updates

    return handler
