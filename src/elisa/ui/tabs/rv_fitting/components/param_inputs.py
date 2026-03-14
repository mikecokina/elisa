"""Gradio component builder for RV fitting initial parameters.

Each fitting parameter exposes four controls:

- ``{name}_value`` - initial (or central) value
- ``{name}_fixed`` - whether the parameter is held fixed
- ``{name}_min``   - lower bound (ignored when fixed)
- ``{name}_max``   - upper bound (ignored when fixed)

:data:`PARAMS` lists the parameter names in the order they appear in the form.
:data:`FIELD_ORDER` is the flat tuple used as Gradio input ordering:
``(p0_value, p0_fixed, p0_min, p0_max, p1_value, ...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from elisa.types import Float

# ---------------------------------------------------------------------------
# Parameter catalogue
# ---------------------------------------------------------------------------

# Parameters for the community RV method (mass_ratio + asini approach).
# Each entry: name -> (label, default_value, default_fixed, default_min, default_max, unit_hint)
# unit_hint is shown in the label only; actual unit strings used in compute.py.
_PARAM_SPEC: dict[str, tuple[str, float, bool, float | None, float | None, str]] = {
    "eccentricity": (
        "Eccentricity  e",
        0.03, False, 0.0, 0.1, "",
    ),
    "asini": (
        "a·sin(i)  [R☉]",
        12.0, False, 8.0, 15.0, "solRad",
    ),
    "mass_ratio": (
        "Mass ratio  q = M₂/M₁",
        1.0, False, 0.9, 1.2, "",
    ),
    "argument_of_periastron": (
        "Arg. of periastron  ω  [deg]",
        170.0, False, 0.0, 360.0, "deg",
    ),
    "gamma": (
        "Systemic velocity  gamma  [km/s]",
        -25.0, False, -50.0, 0.0, "km / s",
    ),
    "period": (
        "Orbital period  P  [d]",
        2.47028, True, 0.01, 1000.0, "d",
    ),
    "primary_minimum_time": (
        "Primary minimum time  T₀  [d]",
        54953.900507000006, True, 2400000.0, 2500000.0, "d",
    ),
    "ln_f": (
        "log noise  ln(f)  [nuisance]",
        -5.0, False, -10.0, 0.0, "",
    ),
}

# Ordered parameter names - defines the row order in the form.
PARAMS: tuple[str, ...] = tuple(_PARAM_SPEC.keys())

# Flat Gradio input ordering: value, fixed, min, max for every parameter.
FIELD_ORDER: tuple[str, ...] = tuple(
    f"{name}_{sub}"
    for name in PARAMS
    for sub in ("value", "fixed", "min", "max")
)


def build(
    *,
    defaults: dict[str, Float | bool | None] | None = None,
) -> dict[str, gr.Component]:
    """Render the initial-parameters section and return a component mapping.

    Creates a header row with column labels and then one row per fitting
    parameter showing its label and four controls (value, fixed, min, max).
    When *defaults* is supplied the matching entries override the built-in
    defaults; unrecognised keys are silently ignored.

    :param defaults: Optional flat mapping of ``"{name}_value"``,
        ``"{name}_fixed"``, ``"{name}_min"``, ``"{name}_max"`` keys to
        override default values.
    :type defaults: dict[str, Float | bool | None] | None
    :returns: Dict keyed by :data:`FIELD_ORDER` entries.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    components: dict[str, gr.Component] = {}

    gr.Markdown("### Initial Parameters")
    gr.Markdown(
        "Set the starting point and search bounds for each parameter.  "
        "Tick **Fixed** to hold a parameter at its value during fitting.",
    )

    # Column header row
    with gr.Row():
        with gr.Column(scale=3, min_width=200):
            gr.Markdown("**Parameter**")
        with gr.Column(scale=2, min_width=120):
            gr.Markdown("**Initial value**")
        with gr.Column(scale=1, min_width=80):
            gr.Markdown("**Fixed?**")
        with gr.Column(scale=2, min_width=100):
            gr.Markdown("**Min**")
        with gr.Column(scale=2, min_width=100):
            gr.Markdown("**Max**")

    for idx, (name, (label, def_val, def_fixed, def_min, def_max, _unit)) in enumerate(_PARAM_SPEC.items()):
        # Add separator before MCMC nuisance parameters
        if name == "ln_f":
            gr.Markdown("---")
            gr.Markdown("**MCMC nuisance parameter**")
            gr.Markdown(
                "Log noise parameter used in MCMC fitting to account for "
                "underestimated observational uncertainties.",
            )
        val = defaults.get(f"{name}_value", def_val)
        fixed = defaults.get(f"{name}_fixed", def_fixed)
        lo = defaults.get(f"{name}_min", def_min)
        hi = defaults.get(f"{name}_max", def_max)

        with gr.Row():
            with gr.Column(scale=3, min_width=200):
                gr.Markdown(label)
            components[f"{name}_value"] = gr.Number(
                value=val,
                label="",
                scale=2,
                show_label=False,
                container=False,
            )
            components[f"{name}_fixed"] = gr.Checkbox(
                value=fixed,
                label="",
                scale=1,
                show_label=False,
                container=False,
            )
            components[f"{name}_min"] = gr.Number(
                value=lo,
                label="",
                scale=2,
                show_label=False,
                container=False,
                interactive=not bool(fixed),
            )
            components[f"{name}_max"] = gr.Number(
                value=hi,
                label="",
                scale=2,
                show_label=False,
                container=False,
                interactive=not bool(fixed),
            )

    return components

