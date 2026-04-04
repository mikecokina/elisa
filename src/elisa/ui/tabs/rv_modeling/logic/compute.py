"""Core computation logic for synthetic radial velocity generation."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from elisa import Observer
from elisa.ui.shared.plotting import render_rv_figure
from elisa.ui.shared.utils import opt_float
from elisa.ui.tabs.lc_modeling.logic.compute import build_star, build_system
from elisa.utc import UTC

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _format_rv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format display precision of phase and RV columns.

    Rounds phase to 4 decimal places and radial velocities to 2 decimal places.

    :param df: Raw DataFrame with ``phase`` and per-component RV columns.
    :type df: pandas.DataFrame;
    :returns: DataFrame with formatted values.
    :rtype: pandas.DataFrame
    """
    out = df.copy()
    out["phase"] = out["phase"].round(4)
    rv_cols = [c for c in out.columns if c != "phase"]
    out[rv_cols] = out[rv_cols].round(2)
    return out


def _save_rv_csv(df: pd.DataFrame) -> str:
    """Save RV DataFrame to a datetime-stamped CSV in the system temp directory.

    The filename has the form ``elisa_rv_YYYY-MM-DD_HH-MM-SS.csv`` so
    successive downloads from the same session are distinct and easy to
    identify.

    :param df: Raw (unrounded) DataFrame to export - full float precision
        is preserved so the downloaded file is suitable for further analysis.
    :type df: pandas.DataFrame
    :returns: Absolute path of the written CSV file.
    :rtype: str
    """
    ts = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(tempfile.gettempdir()) / f"elisa_rv_{ts}.csv"
    df.to_csv(path, index=False)
    return str(path)


def run_rv(
    primary_params: dict[str, object],
    secondary_params: dict[str, object],
    system_params: dict[str, object],
    observer_params: dict[str, object],
) -> tuple[Figure, pd.DataFrame, str]:
    """Compute synthetic RV curves and return figure, table, and CSV path.

    Builds the full ELISa model from the supplied parameter dictionaries,
    runs the radial-velocity synthesis for both components, and returns
    the rendered Matplotlib figure together with a ``pandas.DataFrame``
    containing phases and per-component radial velocity columns.

    :param primary_params: Parameters for the primary :class:`~elisa.base.star.Star`.
    :type primary_params: dict[str, object]
    :param secondary_params: Parameters for the secondary :class:`~elisa.base.star.Star`.
    :type secondary_params: dict[str, object]
    :param system_params: Parameters for the :class:`~elisa.binary_system.system.BinarySystem`.
    :type system_params: dict[str, object]
    :param observer_params: Observer and RV sampling parameters, including
        ``from_phase``, ``to_phase``, ``phase_step``, and ``method``
        (``"kinematic"`` or ``"radiometric"``).
    :type observer_params: dict[str, object]
    :returns: A tuple of ``(figure, dataframe, csv_path)`` where *figure* is a
        Matplotlib figure suitable for ``gr.Plot``, *dataframe* contains
        columns ``phase`` and one column per component (``"primary"``,
        ``"secondary"``), and *csv_path* is the absolute path of the exported
        CSV file with a datetime-stamped name.
    :rtype: tuple[matplotlib.figure.Figure, pandas.DataFrame, str]
    :raises ValueError: If required parameters are missing or logically invalid.
    """
    from_phase_raw = opt_float(observer_params.get("from_phase"))
    to_phase_raw = opt_float(observer_params.get("to_phase"))
    phase_step_raw = opt_float(observer_params.get("phase_step"))
    from_phase = from_phase_raw if from_phase_raw is not None else -0.6
    to_phase = to_phase_raw if to_phase_raw is not None else 0.6
    phase_step = phase_step_raw if phase_step_raw is not None else 0.01
    method: str | None = observer_params.get("method") or None
    primary = build_star(primary_params, label="primary")
    secondary = build_star(secondary_params, label="secondary")
    bs = build_system(primary, secondary, system_params)
    observer = Observer(passband=[], system=bs)
    phases, rvs = observer.observe.rv(
        from_phase=from_phase,
        to_phase=to_phase,
        phase_step=phase_step,
        method=method,
    )

    fig = render_rv_figure(phases, rvs)
    df = pd.DataFrame({"phase": phases})
    for component, rv in rvs.items():
        df[component] = rv
    formatted = _format_rv_df(df)
    return fig, formatted, _save_rv_csv(df)
