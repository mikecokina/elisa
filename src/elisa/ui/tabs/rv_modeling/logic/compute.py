"""Core computation logic for synthetic radial velocity generation."""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from elisa import Observer
from elisa.ui.shared.plotting import render_rv_figure

# noinspection PyProtectedMember
from elisa.ui.tabs.lc_modeling.logic.compute import (
    _build_star,
    _build_system,
    _opt_float,
)
from elisa.utc import UTC

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _format_rv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format display precision of phase and RV columns."""
    out = df.copy()
    out["phase"] = out["phase"].round(4)
    rv_cols = [c for c in out.columns if c != "phase"]
    out[rv_cols] = out[rv_cols].round(2)
    return out
def _save_rv_csv(df: pd.DataFrame) -> str:
    """Save RV DataFrame to datetime-stamped CSV."""
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
    """Compute synthetic RV curves and return figure, table, CSV path."""
    from_phase_raw = _opt_float(observer_params.get("from_phase"))
    to_phase_raw = _opt_float(observer_params.get("to_phase"))
    phase_step_raw = _opt_float(observer_params.get("phase_step"))
    from_phase = from_phase_raw if from_phase_raw is not None else -0.6
    to_phase = to_phase_raw if to_phase_raw is not None else 0.6
    phase_step = phase_step_raw if phase_step_raw is not None else 0.01
    method: str | None = observer_params.get("method") or None
    primary = _build_star(primary_params, label="primary")
    secondary = _build_star(secondary_params, label="secondary")
    bs = _build_system(primary, secondary, system_params)
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
