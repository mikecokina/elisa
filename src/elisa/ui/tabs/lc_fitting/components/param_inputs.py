"""Gradio component builder for LC fitting initial parameters.

Three sections are rendered:

- **System** - orbital and geometric parameters; ``semi_major_axis`` has
  a special three-mode selector (free / fixed / constrained).
- **Primary** - primary component physical parameters.
- **Secondary** - secondary component physical parameters (same fields).
- **Nuisance** - ``ln_f`` log-noise term used by MCMC.

Each regular parameter exposes four controls::

    {section}_{name}_value   - initial / central value
    {section}_{name}_fixed   - hold parameter fixed during fitting
    {section}_{name}_min     - lower bound (ignored when fixed)
    {section}_{name}_max     - upper bound (ignored when fixed)

``semi_major_axis`` exposes five controls::

    system_semi_major_axis_value      - initial value
    system_semi_major_axis_mode       - "free" | "fixed" | "constrained"
    system_semi_major_axis_min        - lower bound (active when mode="free")
    system_semi_major_axis_max        - upper bound (active when mode="free")
    system_semi_major_axis_constraint - expression string (visible when mode="constrained")

:data:`SYSTEM_REGULAR_PARAMS`, :data:`COMPONENT_PARAMS` are exported so
``tab.py`` can iterate over them when wiring the fixed-checkbox handlers.

:data:`FIELD_ORDER` is the canonical flat tuple used as Gradio input ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from elisa.types import Float

# ---------------------------------------------------------------------------
# Parameter catalogue
# ---------------------------------------------------------------------------

# (label, default_value, default_fixed, default_min, default_max, unit_hint)
_Spec = tuple[str, float, bool, float | None, float | None, str | None]

_SYSTEM_SPEC: dict[str, _Spec] = {
    "inclination": (
        "**Inclination**  i  [deg]",
        85.0, False, 80.0, 90.0, "deg",
    ),
    "eccentricity": (
        "**Eccentricity**  e",
        0.0, True, 0.0, 1.0, None,
    ),
    "argument_of_periastron": (
        "**Arg. of periastron**  ω  [deg]",
        0.0, True, 0.0, 360.0, "deg",
    ),
    "mass_ratio": (
        "**Mass ratio**  q = M₂/M₁",
        0.5, True, 0.1, 2.0, None,
    ),
    "period": (
        "**Orbital period**  P  [d]",
        4.5, True, 0.01, 1000.0, "d",
    ),
    "primary_minimum_time": (
        "**Primary minimum time**  T₀  [d]",
        54953.9, True, 2400000.0, 2500000.0, "d",
    ),
    "additional_light": (
        "**Additional light**  l₃",
        0.0, True, 0.0, 0.5, None,
    ),
    "phase_shift": (
        "**Phase shift**  Δφ",
        0.0, True, -0.5, 0.5, None,
    ),
}

_SMA_DEFAULT_VALUE: float = 16.515
_SMA_DEFAULT_MODE: str = "constrained"
_SMA_DEFAULT_MIN: float = 10.0
_SMA_DEFAULT_MAX: float = 30.0
_SMA_DEFAULT_CONSTRAINT: str = "16.515 / sin(radians(system@inclination))"

_COMPONENT_SPEC: dict[str, _Spec] = {
    "t_eff": (
        "**Effective temperature**  T_eff  [K]",
        8307.0, False, 7800.0, 8800.0, "K",
    ),
    "surface_potential": (
        "**Surface potential**  Ω",
        3.0, False, 3.0, 5.0, None,
    ),
    "gravity_darkening": (
        "**Gravity darkening**  β",
        0.32, True, 0.0, 1.0, None,
    ),
    "albedo": (
        "**Albedo**  A",
        0.6, True, 0.0, 1.0, None,
    ),
    "synchronicity": (
        "**Synchronicity**  F",
        1.0, True, 0.1, 10.0, None,
    ),
    "metallicity": (
        "**Metallicity**  [Fe/H]",
        0.0, True, -2.0, 1.0, None,
    ),
}

# ---------------------------------------------------------------------------
# Exported name tuples
# ---------------------------------------------------------------------------

#: Regular system param names (excluding semi_major_axis).
SYSTEM_REGULAR_PARAMS: tuple[str, ...] = tuple(_SYSTEM_SPEC.keys())

#: All system param names (includes semi_major_axis).
SYSTEM_PARAMS: tuple[str, ...] = (*SYSTEM_REGULAR_PARAMS, "semi_major_axis")

#: Per-component param names (identical for primary and secondary).
COMPONENT_PARAMS: tuple[str, ...] = tuple(_COMPONENT_SPEC.keys())

# Flat Gradio input ordering - each param has mode, value, constraint, min, max.
FIELD_ORDER: tuple[str, ...] = (
    *(
        f"system_{name}_{sub}"
        for name in SYSTEM_PARAMS
        for sub in ("value", "mode", "constraint", "min", "max")
    ),
    *(
        f"primary_{name}_{sub}"
        for name in COMPONENT_PARAMS
        for sub in ("value", "mode", "constraint", "min", "max")
    ),
    *(
        f"secondary_{name}_{sub}"
        for name in COMPONENT_PARAMS
        for sub in ("value", "mode", "constraint", "min", "max")
    ),
    "nuisance_ln_f_value",
    "nuisance_ln_f_mode",
    "nuisance_ln_f_constraint",
    "nuisance_ln_f_min",
    "nuisance_ln_f_max",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _param_row(
    components: dict[str, gr.Component],
    section: str,
    name: str,
    label: str,
    value: object,
    fixed: object,
    lo: object,
    hi: object,
    constraint: str | None = None,
) -> None:
    """Render one parameter row with mode selector (free/fixed/constrained).

    Uses the clean semi_major_axis UI pattern: two rows with mode selector,
    conditional constraint field, and interactive min/max bounds.

    :param components: Mutable dict that receives the new components.
    :type components: dict[str, gr.Component]
    :param section: Section name prefix (``"system"``, ``"primary"``, etc.).
    :type section: str
    :param name: Parameter name.
    :type name: str
    :param label: Human-readable label shown in the UI.
    :type label: str
    :param value: Initial value.
    :type value: object
    :param fixed: Initial fixed state.
    :type fixed: object
    :param lo: Initial minimum bound.
    :type lo: object
    :param hi: Initial maximum bound.
    :type hi: object
    :param constraint: Initial constraint expression (if provided).
    :type constraint: str | None
    """
    # Determine mode from constraint/fixed
    if constraint:
        mode = "constrained"
    elif bool(fixed):
        mode = "fixed"
    else:
        mode = "free"

    fv: float | None = float(value) if value is not None else None  # type: ignore[arg-type]
    flo: float | None = float(lo) if lo is not None else None  # type: ignore[arg-type]
    fhi: float | None = float(hi) if hi is not None else None  # type: ignore[arg-type]

    is_free = mode == "free"
    is_constrained = mode == "constrained"

    with gr.Row():
        with gr.Column(scale=3, min_width=200):
            gr.Markdown(label)
        components[f"{section}_{name}_value"] = gr.Number(
            value=fv,
            label="Initial value",
            scale=2,
            container=True,
        )
        components[f"{section}_{name}_mode"] = gr.Radio(
            choices=["free", "fixed", "constrained"],
            value=mode,
            label="Mode",
            scale=3,
        )

    with gr.Row():
        components[f"{section}_{name}_constraint"] = gr.Textbox(
            value=str(constraint) if constraint else "",
            label="Constraint expression",
            info='Expression evaluated after each step, e.g. "16.5 / sin(radians(system@inclination))".',
            scale=6,
            visible=is_constrained,
            container=True,
        )
        components[f"{section}_{name}_min"] = gr.Number(
            value=flo,
            label="Min",
            scale=2,
            interactive=is_free,
            container=True,
        )
        components[f"{section}_{name}_max"] = gr.Number(
            value=fhi,
            label="Max",
            scale=2,
            interactive=is_free,
            container=True,
        )


def _build_component_section(
    components: dict[str, gr.Component],
    section: str,
    spec: dict[str, _Spec],
    defaults: dict[str, Float | bool | str | None],
    value_overrides: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Render one primary/secondary component section.

    :param components: Mutable dict that receives the new components.
    :type components: dict[str, gr.Component]
    :param section: Section prefix (``"primary"`` or ``"secondary"``).
    :type section: str
    :param spec: Parameter spec dict (label, default_val, default_fixed, min, max, unit).
    :type spec: dict[str, _Spec]
    :param defaults: User-supplied defaults override dict.
    :type defaults: dict[str, Float | bool | str | None]
    :param value_overrides: Optional per-name ``(value, min, max)`` overrides used
        for the secondary component whose default values differ from the primary.
    :type value_overrides: dict[str, tuple[float, float, float]] | None
    """
    for name, (label, def_val, def_fixed, def_min, def_max, _unit) in spec.items():
        if value_overrides and name in value_overrides:
            sec_dv, sec_dlo, sec_dhi = value_overrides[name]
            val = defaults.get(f"{section}_{name}_value", sec_dv)
            lo = defaults.get(f"{section}_{name}_min", sec_dlo)
            hi = defaults.get(f"{section}_{name}_max", sec_dhi)
        else:
            val = defaults.get(f"{section}_{name}_value", def_val)
            lo = defaults.get(f"{section}_{name}_min", def_min)
            hi = defaults.get(f"{section}_{name}_max", def_max)
        fixed = defaults.get(f"{section}_{name}_fixed", def_fixed)
        constraint = defaults.get(f"{section}_{name}_constraint")
        _param_row(components, section, name, label, val, fixed, lo, hi, constraint)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build(
    *,
    defaults: dict[str, Float | bool | str | None] | None = None,
) -> dict[str, gr.Component]:
    """Render the initial-parameters form and return a component mapping.

    Creates three labelled sections (System, Primary, Secondary) plus a
    nuisance row for MCMC.  An optional *defaults* dict overrides the
    built-in starting values; unrecognised keys are silently ignored.

    The ``semi_major_axis`` row exposes a three-way mode radio
    (``"free"`` / ``"fixed"`` / ``"constrained"``).  When the mode is
    ``"constrained"`` a constraint-expression textbox is shown; when
    ``"free"`` the min/max fields are interactive.  Visibility and
    interactivity toggling is wired in ``tab.py`` via the
    ``system_semi_major_axis_mode`` component.

    :param defaults: Flat mapping of ``"{section}_{name}_{sub}"`` keys
        to override default values.  For ``semi_major_axis``, additional
        keys ``system_semi_major_axis_mode`` and
        ``system_semi_major_axis_constraint`` are recognised.
    :type defaults: dict[str, Float | bool | str | None] | None
    :returns: Dict keyed by every entry in :data:`FIELD_ORDER`.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    components: dict[str, gr.Component] = {}

    # ------------------------------------------------------------------ #
    # Section: System                                                       #
    # ------------------------------------------------------------------ #
    with gr.Accordion("System Parameters", open=True):
        for name, (label, def_val, def_fixed, def_min, def_max, _unit) in _SYSTEM_SPEC.items():
            val = defaults.get(f"system_{name}_value", def_val)
            fixed = defaults.get(f"system_{name}_fixed", def_fixed)
            lo = defaults.get(f"system_{name}_min", def_min)
            hi = defaults.get(f"system_{name}_max", def_max)
            constraint = defaults.get(f"system_{name}_constraint")
            _param_row(components, "system", name, label, val, fixed, lo, hi, constraint)

        # -- semi_major_axis (now uses same pattern as other params) --
        sma_val = defaults.get("system_semi_major_axis_value", _SMA_DEFAULT_VALUE)
        sma_mode_str = defaults.get("system_semi_major_axis_mode", _SMA_DEFAULT_MODE)
        sma_fixed = sma_mode_str == "fixed" if isinstance(sma_mode_str, str) else False
        sma_lo = defaults.get("system_semi_major_axis_min", _SMA_DEFAULT_MIN)
        sma_hi = defaults.get("system_semi_major_axis_max", _SMA_DEFAULT_MAX)
        sma_constraint = defaults.get("system_semi_major_axis_constraint", _SMA_DEFAULT_CONSTRAINT)
        sma_constraint = (sma_constraint or _SMA_DEFAULT_CONSTRAINT) if sma_mode_str == "constrained" else None

        _param_row(
            components, "system", "semi_major_axis",
            "**Semi-major axis**  a  [R☉]",
            sma_val, sma_fixed, sma_lo, sma_hi, sma_constraint,
        )

    # ------------------------------------------------------------------ #
    # Section: Primary                                                      #
    # ------------------------------------------------------------------ #
    with gr.Accordion("Primary Component Parameters", open=False):
        _build_component_section(components, "primary", _COMPONENT_SPEC, defaults)

    # ------------------------------------------------------------------ #
    # Section: Secondary                                                    #
    # ------------------------------------------------------------------ #
    with gr.Accordion("Secondary Component Parameters", open=False):
        _secondary_value_overrides: dict[str, tuple[float, float, float]] = {
            "t_eff": (4000.0, 4000.0, 7000.0),
            "surface_potential": (5.0, 5.0, 7.0),
            "gravity_darkening": (0.32, 0.0, 1.0),
            "albedo": (0.6, 0.0, 1.0),
            "synchronicity": (1.0, 0.1, 10.0),
            "metallicity": (0.0, -2.0, 1.0),
        }
        _build_component_section(
            components, "secondary", _COMPONENT_SPEC, defaults,
            value_overrides=_secondary_value_overrides,
        )

    # ------------------------------------------------------------------ #
    # MCMC nuisance                                                         #
    # ------------------------------------------------------------------ #
    with gr.Accordion("MCMC Nuisance Parameter", open=False):
        gr.Markdown("Log noise term that accounts for underestimated observational uncertainties.")

        ln_f_val = defaults.get("nuisance_ln_f_value", -5.0)
        ln_f_fixed = bool(defaults.get("nuisance_ln_f_fixed", False))
        ln_f_lo = defaults.get("nuisance_ln_f_min", -10.0)
        ln_f_hi = defaults.get("nuisance_ln_f_max", 0.0)
        ln_f_constraint = defaults.get("nuisance_ln_f_constraint")

        _param_row(
            components, "nuisance", "ln_f",
            "**Log noise**  ln(f)  [nuisance]",
            ln_f_val, ln_f_fixed, ln_f_lo, ln_f_hi, ln_f_constraint,
        )

    return components

