"""Gradio component builder for RV fitting initial parameters.

Each fitting parameter exposes five controls::

    {name}_value      - initial / central value
    {name}_mode       - "free" | "fixed" | "constrained"
    {name}_constraint - constraint expression (active when mode="constrained")
    {name}_min        - lower bound (active when mode="free")
    {name}_max        - upper bound (active when mode="free")

:data:`PARAMS` lists the parameter names in the order they appear in the form.
:data:`FIELD_ORDER` is the flat tuple used as Gradio input ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gradio as gr

if TYPE_CHECKING:
    from elisa.types import Float

# ---------------------------------------------------------------------------
# Parameter catalogue
# ---------------------------------------------------------------------------

# (label, default_value, default_fixed, default_min, default_max, unit_hint)
_Spec = tuple[str, float, bool, float | None, float | None, str]

PARAM_SPEC: dict[str, _Spec] = {
    "eccentricity": (
        "**Eccentricity**  e",
        0.03, False, 0.0, 0.1, "",
    ),
    "asini": (
        "**a·sin(i)**  [R☉]",
        12.0, False, 8.0, 15.0, "solRad",
    ),
    "mass_ratio": (
        "**Mass ratio**  q = M₂/M₁",
        1.0, False, 0.9, 1.2, "",
    ),
    "argument_of_periastron": (
        "**Arg. of periastron**  ω  [deg]",
        170.0, False, 0.0, 360.0, "deg",
    ),
    "gamma": (
        "**Systemic velocity**  gamma  [km/s]",
        -25.0, False, -50.0, 0.0, "km / s",
    ),
    "period": (
        "**Orbital period**  P  [d]",
        2.47028, True, 0.01, 1000.0, "d",
    ),
    "primary_minimum_time": (
        "**Primary minimum time**  T₀  [d]",
        54953.900507000006, True, 2400000.0, 2500000.0, "d",
    ),
    "ln_f": (
        "**Log noise**  ln(f)  [nuisance]",
        -5.0, False, -10.0, 0.0, "",
    ),
}

# Ordered parameter names - defines the row order in the form.
PARAMS: tuple[str, ...] = tuple(PARAM_SPEC.keys())

# Flat Gradio input ordering: value, mode, constraint, min, max for every parameter.
FIELD_ORDER: tuple[str, ...] = tuple(
    f"{name}_{sub}"
    for name in PARAMS
    for sub in ("value", "mode", "constraint", "min", "max")
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mode_handler(mode: str) -> tuple[Any, Any, Any]:
    """Handle mode radio change for any parameter row.

    - **free** - min and max editable, constraint greyed out.
    - **fixed** - min, max, and constraint all greyed out.
    - **constrained** - constraint editable, min and max greyed out.

    :param mode: Selected mode value (``"free"``, ``"fixed"``, or ``"constrained"``).
    :type mode: str
    :returns: Tuple of ``gr.update`` objects for (constraint, min, max).
    :rtype: tuple[Any, Any, Any]
    """
    if mode == "free":
        return (
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    if mode == "fixed":
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    # constrained
    return (
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def _param_row(
    components: dict[str, gr.Component],
    name: str,
    label: str,
    value: float,
    lo: float | None,
    hi: float | None,
    *,
    fixed: bool = False,
) -> None:
    """Render one parameter row with mode selector (free/fixed/constrained).

    All secondary fields (constraint, min, max) are always present in the DOM.
    Interactivity is toggled by the mode radio to avoid Gradio's
    loading-spinner bug with ``visible`` updates inside Tabs/Accordions.

    :param components: Mutable dict that receives the new components.
    :type components: dict[str, gr.Component]
    :param name: Parameter name used as key prefix.
    :type name: str
    :param label: Human-readable label (Markdown supported).
    :type label: str
    :param value: Default initial value.
    :type value: float
    :param fixed: Whether the parameter defaults to fixed mode.
    :type fixed: bool
    :param lo: Default lower bound.
    :type lo: float | None
    :param hi: Default upper bound.
    :type hi: float | None
    """
    mode = "fixed" if fixed else "free"
    bounds_active = not fixed

    with gr.Row():
        with gr.Column(scale=3, min_width=200):
            gr.Markdown(label)
        value_comp = gr.Number(
            value=value,
            label="Initial value",
            scale=2,
            interactive=True,
            container=True,
        )
        mode_comp = gr.Radio(
            choices=["free", "fixed", "constrained"],
            value=mode,
            label="Mode",
            scale=3,
            interactive=True,
        )

    with gr.Row():
        constraint_comp = gr.Textbox(
            value="",
            label="Constraint expression",
            placeholder='e.g. "(1 + system@eccentricity)**2 / (1 - system@eccentricity**2)**(3.0/2.0)"',
            scale=4,
            interactive=False,
            container=True,
        )
        min_comp = gr.Number(
            value=lo,
            label="Min",
            scale=2,
            interactive=bounds_active,
            container=True,
        )
        max_comp = gr.Number(
            value=hi,
            label="Max",
            scale=2,
            interactive=bounds_active,
            container=True,
        )

    mode_comp.change(
        fn=_mode_handler,
        inputs=[mode_comp],
        outputs=[constraint_comp, min_comp, max_comp],
    )

    components[f"{name}_value"] = value_comp
    components[f"{name}_mode"] = mode_comp
    components[f"{name}_constraint"] = constraint_comp
    components[f"{name}_min"] = min_comp
    components[f"{name}_max"] = max_comp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build(
    *,
    defaults: dict[str, Float | bool | None] | None = None,
) -> dict[str, gr.Component]:
    """Render the initial-parameters section and return a component mapping.

    Creates one collapsible section for system parameters and one for the MCMC
    nuisance parameter.  An optional *defaults* dict overrides the built-in
    starting values; unrecognised keys are silently ignored.

    :param defaults: Optional flat mapping of ``"{name}_value"``,
        ``"{name}_mode"``, ``"{name}_constraint"``, ``"{name}_min"``,
        ``"{name}_max"`` keys to override default values.
    :type defaults: dict[str, Float | bool | None] | None
    :returns: Dict keyed by :data:`FIELD_ORDER` entries.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    components: dict[str, gr.Component] = {}

    with gr.Accordion("System Parameters", open=True):
        for name, (label, def_val, def_fixed, def_min, def_max, _unit) in PARAM_SPEC.items():
            if name == "ln_f":
                continue
            mode_str = str(defaults.get(f"{name}_mode", "fixed" if def_fixed else "free"))
            val = float(defaults.get(f"{name}_value", def_val))  # type: ignore[arg-type]
            lo = defaults.get(f"{name}_min", def_min)
            hi = defaults.get(f"{name}_max", def_max)
            _param_row(components, name, label, val, lo, hi, fixed=mode_str == "fixed")  # type: ignore[arg-type]

    with gr.Accordion("MCMC Nuisance Parameter", open=False):
        gr.Markdown(
            "Log noise parameter used in MCMC fitting to account for "
            "underestimated observational uncertainties.",
        )
        label, def_val, def_fixed, def_min, def_max, _unit = PARAM_SPEC["ln_f"]
        mode_str = str(defaults.get("ln_f_mode", "fixed" if def_fixed else "free"))
        val = float(defaults.get("ln_f_value", def_val))  # type: ignore[arg-type]
        lo = defaults.get("ln_f_min", def_min)
        hi = defaults.get("ln_f_max", def_max)
        _param_row(components, "ln_f", label, val, lo, hi, fixed=mode_str == "fixed")  # type: ignore[arg-type]

    return components

