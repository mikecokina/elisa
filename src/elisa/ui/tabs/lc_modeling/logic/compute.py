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

from elisa import Observer
from elisa.ui.shared.binary_model import build_star, build_system
from elisa.ui.shared.fit_json import load_model_params_from_json  # noqa: F401 - re-exported for callers
from elisa.ui.shared.utils import opt_float
from elisa.utc import UTC

if TYPE_CHECKING:
    from matplotlib.figure import Figure



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
