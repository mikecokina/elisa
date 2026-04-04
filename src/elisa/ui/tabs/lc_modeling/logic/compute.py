"""Core computation logic for synthetic light curve generation.

This module is a pure-logic layer with no Gradio dependency. It translates
raw parameter dictionaries (as produced by the Gradio form) into ELISa
objects, runs the light-curve synthesis, and returns results that can be
rendered by the UI layer.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from elisa import BinarySystem, Observer, Star
from elisa import units as u
from elisa.ui.shared.fit_json import load_model_params_from_json  # noqa: F401 - re-exported for callers
from elisa.ui.shared.utils import opt_float
from elisa.utc import UTC

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from elisa.types import Float, Int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _opt_int(value: Float | str | None) -> Int | None:
    """Return an int if *value* is a non-empty, non-``None`` value.

    Accepts plain numbers and strings produced by ``gr.Textbox`` for
    optional parameters.  An empty string or ``None`` both signal
    "not supplied" and return ``None``.

    :param value: Numeric value, string representation, or ``None``.
    :type value: Float | str | None
    :returns: Parsed int or ``None``.
    :rtype: Int | None
    :raises ValueError: If *value* is a non-empty string that cannot be
        parsed as a number, or if the parsed value is not a positive integer.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            parsed = float(stripped)
        except ValueError as exc:
            msg = f"Cannot parse '{stripped}' as a number."
            raise ValueError(msg) from exc
        return int(parsed)
    return int(value)


def _validated_float(
    value: Float | str | None,
    *,
    name: str,
    lo: Float | None = None,
    hi: Float | None = None,
) -> Float | None:
    """Parse *value* as an optional float and validate it against a range.

    Returns ``None`` when *value* is empty/``None`` so the caller can
    skip the parameter and let ELISa apply its default.  When a value is
    present it must lie within the half-open interval ``[lo, hi]``
    (boundaries are inclusive when provided).

    :param value: Numeric value, string, or ``None``.
    :type value: Float | str | None
    :param name: Parameter name used in error messages.
    :type name: str
    :param lo: Optional inclusive lower bound.
    :type lo: Float | None
    :param hi: Optional inclusive upper bound.
    :type hi: Float | None
    :returns: Parsed and validated float, or ``None`` if not supplied.
    :rtype: Float | None
    :raises ValueError: If the parsed value lies outside ``[lo, hi]``.
    """
    parsed = opt_float(value)
    if parsed is None:
        return None
    if lo is not None and parsed < lo:
        msg = f"'{name}' must be >= {lo}, got {parsed}."
        raise ValueError(msg)
    if hi is not None and parsed > hi:
        msg = f"'{name}' must be <= {hi}, got {parsed}."
        raise ValueError(msg)
    return parsed


def _validated_positive_int(value: Float | str | None, *, name: str) -> Int | None:
    """Parse *value* as an optional positive integer.

    Returns ``None`` for empty/``None`` input.

    :param value: Numeric value, string, or ``None``.
    :type value: Float | str | None
    :param name: Parameter name used in error messages.
    :type name: str
    :returns: Parsed positive integer, or ``None`` if not supplied.
    :rtype: Int | None
    :raises ValueError: If the parsed value is not a positive integer.
    """
    parsed = _opt_int(value)
    if parsed is None:
        return None
    if parsed <= 0:
        msg = f"'{name}' must be a positive integer, got {parsed}."
        raise ValueError(msg)
    return parsed


def build_star(
    params: dict[str, object],
    *,
    label: str,
    pulsations: list[dict[str, object]] | None = None,
    spots: list[dict[str, object]] | None = None,
) -> Star:
    """Construct a :class:`~elisa.base.star.Star` from a flat parameter dict.

    The dict is expected to contain keys produced by
    :mod:`~elisa.ui.tabs.lc_modeling.components.star_inputs`.

    :param params: Flat parameter dictionary with keys ``mass``,
        ``t_eff``, ``surface_potential``, ``synchronicity``, and
        optional keys ``gravity_darkening``, ``albedo``,
        ``metallicity``, ``discretization_factor``, ``atmosphere``.
    :type params: dict[str, object]
    :param label: Human-readable label used in error messages
        (e.g. ``"primary"`` or ``"secondary"``).
    :type label: str
    :param pulsations: Optional list of pulsation mode parameter dicts,
        each with keys ``l``, ``m``, ``amplitude``, ``frequency``, and
        optional keys accepted by :class:`~elisa.pulse.mode.PulsationMode`.
        Pass ``None`` or an empty list for no pulsations.
    :type pulsations: list[dict[str, object]] | None
    :param spots: Optional list of spot parameter dicts, each with keys
        ``longitude``, ``latitude``, ``angular_radius``, and
        ``temperature_factor``. Pass ``None`` or an empty list for no spots.
    :type spots: list[dict[str, object]] | None
    :returns: Initialised :class:`~elisa.base.star.Star` instance.
    :rtype: Star
    :raises ValueError: If any mandatory parameter is ``None`` or
        missing from *params*.
    """
    # --- mandatory ---
    mass_raw = opt_float(params.get("mass"))  # type: ignore[arg-type]
    t_eff_raw = opt_float(params.get("t_eff"))  # type: ignore[arg-type]
    surface_potential_raw = opt_float(params.get("surface_potential"))  # type: ignore[arg-type]
    synchronicity_raw = opt_float(params.get("synchronicity"))  # type: ignore[arg-type]

    for name, val in [
        ("mass", mass_raw),
        ("t_eff", t_eff_raw),
        ("surface_potential", surface_potential_raw),
        ("synchronicity", synchronicity_raw),
    ]:
        if val is None:
            msg = f"{label} star: mandatory parameter '{name}' is missing or empty."
            raise ValueError(msg)

    kwargs: dict[str, object] = {
        "mass": mass_raw * u.solMass,
        "t_eff": t_eff_raw * u.K,
        "surface_potential": surface_potential_raw,
        "synchronicity": synchronicity_raw,
    }

    # --- optional ---
    gravity_darkening = _validated_float(
        params.get("gravity_darkening"),  # type: ignore[arg-type]
        name=f"{label}.gravity_darkening",
        lo=0.0,
        hi=1.0,
    )
    albedo = _validated_float(
        params.get("albedo"),  # type: ignore[arg-type]
        name=f"{label}.albedo",
        lo=0.0,
        hi=1.0,
    )
    metallicity = opt_float(params.get("metallicity"))  # type: ignore[arg-type]
    discretization_factor = _validated_positive_int(
        params.get("discretization_factor"),  # type: ignore[arg-type]
        name=f"{label}.discretization_factor",
    )
    atmosphere: str | None = params.get("atmosphere") or None  # type: ignore[assignment]

    if gravity_darkening is not None:
        kwargs["gravity_darkening"] = gravity_darkening
    if albedo is not None:
        kwargs["albedo"] = albedo
    if metallicity is not None:
        kwargs["metallicity"] = metallicity
    if discretization_factor is not None:
        kwargs["discretization_factor"] = discretization_factor
    if atmosphere:
        kwargs["atmosphere"] = atmosphere
    if pulsations:
        kwargs["pulsations"] = pulsations
    if spots:
        kwargs["spots"] = spots

    return Star(**kwargs)


def build_system(
    primary: Star,
    secondary: Star,
    params: dict[str, object],
) -> BinarySystem:
    """Construct a :class:`~elisa.binary_system.system.BinarySystem`.

    :param primary: Primary stellar component.
    :type primary: Star
    :param secondary: Secondary stellar component.
    :type secondary: Star
    :param params: Flat parameter dict with keys ``inclination``,
        ``period``, ``eccentricity``, ``argument_of_periastron``,
        and optional keys ``gamma``, ``phase_shift``,
        ``additional_light``, ``primary_minimum_time``, ``distance``.
    :type params: dict[str, object]
    :returns: Initialised :class:`~elisa.binary_system.system.BinarySystem`.
    :rtype: BinarySystem
    :raises ValueError: If any mandatory parameter is ``None`` or missing.
    """
    # --- mandatory ---
    inclination_raw = opt_float(params.get("inclination"))  # type: ignore[arg-type]
    period_raw = opt_float(params.get("period"))  # type: ignore[arg-type]
    eccentricity_raw = opt_float(params.get("eccentricity"))  # type: ignore[arg-type]
    aop_raw = opt_float(params.get("argument_of_periastron"))  # type: ignore[arg-type]

    for name, val in [
        ("inclination", inclination_raw),
        ("period", period_raw),
        ("eccentricity", eccentricity_raw),
        ("argument_of_periastron", aop_raw),
    ]:
        if val is None:
            msg = f"Binary system: mandatory parameter '{name}' is missing or empty."
            raise ValueError(msg)

    bs_kwargs: dict[str, object] = {
        "inclination": inclination_raw * u.deg,
        "period": period_raw * u.d,
        "eccentricity": eccentricity_raw,
        "argument_of_periastron": aop_raw * u.deg,
    }

    # --- optional ---
    gamma_raw = opt_float(params.get("gamma"))  # type: ignore[arg-type]
    phase_shift_raw = opt_float(params.get("phase_shift"))  # type: ignore[arg-type]
    additional_light_raw = _validated_float(
        params.get("additional_light"),  # type: ignore[arg-type]
        name="additional_light",
        lo=0.0,
        hi=1.0,
    )
    pmt_raw = opt_float(params.get("primary_minimum_time"))  # type: ignore[arg-type]
    distance_raw = _validated_float(
        params.get("distance"),  # type: ignore[arg-type]
        name="distance",
        lo=0.0,
    )

    if gamma_raw is not None:
        bs_kwargs["gamma"] = gamma_raw * u.km / u.s
    if phase_shift_raw is not None:
        bs_kwargs["phase_shift"] = phase_shift_raw
    if additional_light_raw is not None:
        bs_kwargs["additional_light"] = additional_light_raw
    if pmt_raw is not None:
        bs_kwargs["primary_minimum_time"] = pmt_raw * u.d
    if distance_raw is not None:
        bs_kwargs["distance"] = distance_raw * u.pc

    return BinarySystem(primary=primary, secondary=secondary, **bs_kwargs)


def _format_df(df: pd.DataFrame, *, normalize: bool) -> pd.DataFrame:
    """Format display precision of phase and flux columns.

    - Phase is rounded to 4 decimal places.
    - Normalised flux is rounded to 5 decimal places.
    - Absolute flux is formatted as compact scientific notation (``1.234e+05``)
      so the table stays readable without losing order-of-magnitude context.

    :param df: Raw DataFrame with ``phase`` and per-passband flux columns.
    :type df: pandas.DataFrame
    :param normalize: Whether the flux columns are normalised (dimensionless).
    :type normalize: bool
    :returns: DataFrame with formatted values.  Absolute-flux columns are
        stored as strings to preserve the ``1.234e+05`` representation.
    :rtype: pandas.DataFrame
    """
    out = df.copy()
    out["phase"] = out["phase"].round(4)

    flux_cols = [c for c in out.columns if c != "phase"]
    if normalize:
        out[flux_cols] = out[flux_cols].round(5)
    else:
        for col in flux_cols:
            out[col] = out[col].apply(lambda x: f"{x:.3e}")
    return out


def _save_lc_csv(df: pd.DataFrame) -> str:
    """Save *df* to a datetime-stamped CSV in the system temp directory.

    The filename has the form ``elisa_lc_YYYY-MM-DD_HH-MM-SS.csv`` so
    successive downloads from the same session are distinct and easy to
    identify.

    :param df: Raw (unrounded) DataFrame to export - full float precision
        is preserved so the downloaded file is suitable for further analysis.
    :type df: pandas.DataFrame
    :returns: Absolute path of the written CSV file.
    :rtype: str
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(tempfile.gettempdir()) / f"elisa_lc_{ts}.csv"
    df.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_lc(
    primary_params: dict[str, object],
    secondary_params: dict[str, object],
    system_params: dict[str, object],
    observer_params: dict[str, object],
    primary_pulsation_params: dict[str, object] | None = None,
    secondary_pulsation_params: dict[str, object] | None = None,
    primary_spot_params: dict[str, object] | None = None,
    secondary_spot_params: dict[str, object] | None = None,
) -> tuple[Figure, pd.DataFrame, str]:
    """Compute a synthetic light curve and return a figure and a data table.

    Builds the full ELISa model from the supplied parameter dictionaries,
    runs the light-curve synthesis for all selected passbands, and returns
    the rendered Matplotlib figure together with a ``pandas.DataFrame``
    containing phases and per-passband flux columns.

    The parameter dictionaries are expected to contain the same keys that
    the corresponding Gradio component builders expose (see the
    ``components`` sub-package).

    :param primary_params: Parameters for the primary :class:`~elisa.base.star.Star`.
    :type primary_params: dict[str, object]
    :param secondary_params: Parameters for the secondary :class:`~elisa.base.star.Star`.
    :type secondary_params: dict[str, object]
    :param system_params: Parameters for the :class:`~elisa.binary_system.system.BinarySystem`.
    :type system_params: dict[str, object]
    :param observer_params: Observer and light-curve sampling parameters,
        including ``passband`` (list of str), ``from_phase``, ``to_phase``,
        ``phase_step``, and ``normalize`` (bool).
    :type observer_params: dict[str, object]
    :param primary_pulsation_params: Primary star pulsation mode parameters.
        When ``None`` or empty, no pulsations are applied to the primary.
    :type primary_pulsation_params: dict[str, object] | None
    :param secondary_pulsation_params: Secondary star pulsation mode parameters.
        When ``None`` or empty, no pulsations are applied to the secondary.
    :type secondary_pulsation_params: dict[str, object] | None
    :param primary_spot_params: Primary star spot parameters. When ``None`` or
        empty, no spots are applied to the primary.
    :type primary_spot_params: dict[str, object] | None
    :param secondary_spot_params: Secondary star spot parameters. When ``None``
        or empty, no spots are applied to the secondary.
    :type secondary_spot_params: dict[str, object] | None
    :returns: A tuple of ``(figure, dataframe, csv_path)`` where *figure* is a
        Matplotlib figure suitable for ``gr.Plot``, *dataframe* contains
        columns ``phase`` and one column per passband, and *csv_path* is the
        absolute path of the exported CSV file with a datetime-stamped name.
    :rtype: tuple[matplotlib.figure.Figure, pandas.DataFrame, str]
    :raises ValueError: If required parameters are missing or logically
        invalid (e.g. no passbands selected).
    """
    from elisa.ui.tabs.lc_modeling.components.pulsation_inputs import (  # noqa: PLC0415
        parse_pulsation_modes,
    )

    # --- validate observer params early for a clear error message ---
    passbands: list[str] = observer_params.get("passband") or []  # type: ignore[assignment]
    if not passbands:
        msg = "At least one passband must be selected."
        raise ValueError(msg)

    from_phase_raw = opt_float(observer_params.get("from_phase"))  # type: ignore[arg-type]
    to_phase_raw = opt_float(observer_params.get("to_phase"))  # type: ignore[arg-type]
    phase_step_raw = opt_float(observer_params.get("phase_step"))  # type: ignore[arg-type]

    from_phase = from_phase_raw if from_phase_raw is not None else -0.6
    to_phase = to_phase_raw if to_phase_raw is not None else 0.6
    phase_step = phase_step_raw if phase_step_raw is not None else 0.01
    normalize: bool = bool(observer_params.get("normalize", False))

    # --- build ELISa objects ---
    prim_pulsations = parse_pulsation_modes(primary_pulsation_params)
    sec_pulsations = parse_pulsation_modes(secondary_pulsation_params)

    from elisa.ui.tabs.lc_modeling.components.spot_inputs import parse_spots  # noqa: PLC0415

    prim_spots = parse_spots(primary_spot_params)
    sec_spots = parse_spots(secondary_spot_params)

    primary = build_star(
        primary_params,
        label="primary",
        pulsations=prim_pulsations or None,
        spots=prim_spots or None,
    )
    secondary = build_star(
        secondary_params,
        label="secondary",
        pulsations=sec_pulsations or None,
        spots=sec_spots or None,
    )
    bs = build_system(primary, secondary, system_params)
    observer = Observer(passband=passbands, system=bs)

    # --- run light-curve synthesis ---
    phases, fluxes = observer.observe.lc(
        from_phase=from_phase,
        to_phase=to_phase,
        phase_step=phase_step,
        normalize=normalize,
    )

    # --- render figure ---
    from elisa.ui.shared.plotting import render_lc_figure  # noqa: PLC0415

    fig = render_lc_figure(phases, fluxes, normalize=normalize)

    # --- build DataFrame ---
    df = pd.DataFrame({"phase": phases})
    for band, flux in fluxes.items():
        df[band] = flux

    formatted = _format_df(df, normalize=normalize)
    return fig, formatted, _save_lc_csv(df)
