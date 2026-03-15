"""Gradio component builder for LC fitting initial parameters.

Supports two fitting approaches:

- **Community** - uses ``semi_major_axis`` and ``mass_ratio`` instead of individual masses.
- **Standard** - uses individual component masses.

Three sections are rendered:

- **System** - orbital and geometric parameters.
- **Primary** - primary component physical parameters.
- **Secondary** - secondary component physical parameters (same fields).
- **Nuisance** - ``ln_f`` log-noise term used by MCMC.

Each regular parameter exposes five controls::

    {section}_{name}_value      - initial / central value
    {section}_{name}_mode       - "free" | "fixed" | "constrained"
    {section}_{name}_constraint - expression string (visible when mode="constrained")
    {section}_{name}_min        - lower bound (active when mode="free")
    {section}_{name}_max        - upper bound (active when mode="free")

:data:`FIELD_ORDER` is the canonical flat tuple used as Gradio input ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import gradio as gr

if TYPE_CHECKING:
    from collections.abc import Callable

    from elisa.types import Float

# ---------------------------------------------------------------------------
# Parameter catalogue
# ---------------------------------------------------------------------------

# (label, default_value, default_fixed, default_min, default_max, unit_hint)
_Spec = tuple[str, float, bool, float | None, float | None, str | None]

# Common system parameters for both approaches
_SYSTEM_COMMON_SPEC: dict[str, _Spec] = {
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

# Community-specific system parameters
_SYSTEM_COMMUNITY_SPEC: dict[str, _Spec] = {
    "semi_major_axis": (
        "**Semi-major axis**  a  [R☉]",
        16.515, False, 10.0, 30.0, "R☉",
    ),
    "mass_ratio": (
        "**Mass ratio**  q = M₂/M₁",
        0.5, True, 0.1, 2.0, None,
    ),
}

# Component parameters common to both approaches
_COMPONENT_COMMON_SPEC: dict[str, _Spec] = {
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

# Standard-specific component parameter
_COMPONENT_STANDARD_SPEC: dict[str, _Spec] = {
    "mass": (
        "**Mass**  M  [M☉]",
        2.0, False, 0.1, 100.0, "M☉",
    ),
}

# ---------------------------------------------------------------------------
# Exported name tuples
# ---------------------------------------------------------------------------

#: System params for community approach (common + semi_major_axis + mass_ratio).
SYSTEM_PARAMS_COMMUNITY: tuple[str, ...] = (
    *tuple(_SYSTEM_COMMON_SPEC.keys()),
    *tuple(_SYSTEM_COMMUNITY_SPEC.keys()),
)

#: System params for standard approach (common only, no semi_major_axis or mass_ratio).
SYSTEM_PARAMS_STANDARD: tuple[str, ...] = tuple(_SYSTEM_COMMON_SPEC.keys())

#: Component params for community approach (no mass).
COMPONENT_PARAMS_COMMUNITY: tuple[str, ...] = tuple(_COMPONENT_COMMON_SPEC.keys())

#: Component params for standard approach (mass + common).
COMPONENT_PARAMS_STANDARD: tuple[str, ...] = (
    *tuple(_COMPONENT_STANDARD_SPEC.keys()),
    *tuple(_COMPONENT_COMMON_SPEC.keys()),
)

# Backward compatibility exports - default to community approach
SYSTEM_PARAMS: tuple[str, ...] = SYSTEM_PARAMS_COMMUNITY
SYSTEM_REGULAR_PARAMS: tuple[str, ...] = tuple(_SYSTEM_COMMON_SPEC.keys())
COMPONENT_PARAMS: tuple[str, ...] = COMPONENT_PARAMS_COMMUNITY


def _build_field_order(approach: Literal["community", "standard"]) -> tuple[str, ...]:
    """Build FIELD_ORDER dynamically based on the approach.

    :param approach: Fitting approach ("community" or "standard").
    :type approach: Literal["community", "standard"]
    :returns: Tuple of field names in canonical order.
    :rtype: tuple[str, ...]
    """
    system_params = SYSTEM_PARAMS_COMMUNITY if approach == "community" else SYSTEM_PARAMS_STANDARD
    component_params = COMPONENT_PARAMS_COMMUNITY if approach == "community" else COMPONENT_PARAMS_STANDARD

    return (
        *(
            f"system_{name}_{sub}"
            for name in system_params
            for sub in ("value", "mode", "constraint", "min", "max")
        ),
        *(
            f"primary_{name}_{sub}"
            for name in component_params
            for sub in ("value", "mode", "constraint", "min", "max")
        ),
        *(
            f"secondary_{name}_{sub}"
            for name in component_params
            for sub in ("value", "mode", "constraint", "min", "max")
        ),
        "nuisance_ln_f_value",
        "nuisance_ln_f_mode",
        "nuisance_ln_f_constraint",
        "nuisance_ln_f_min",
        "nuisance_ln_f_max",
    )


# Default to community approach for backward compatibility
FIELD_ORDER: tuple[str, ...] = _build_field_order("community")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_mode_change_handler() -> Callable[[str], tuple[Any, Any, Any]]:
    """Create a Gradio update handler for parameter mode changes.

    Returns a callable that takes mode string and returns tuple of update
    objects for (constraint, min, max) components.

    :returns: Callable that takes mode string and returns tuple of gr.update()s.
    :rtype: Callable[[str], tuple[Any, Any, Any]]
    """
    def handler(mode: str) -> tuple[Any, Any, Any]:
        """Handle mode change for a parameter.

        - **free**: show min/max, hide constraint
        - **fixed**: hide min/max and constraint
        - **constrained**: show constraint, hide min/max

        :param mode: New mode value ("free", "fixed", or "constrained").
        :type mode: str
        :returns: Tuple of updates for constraint, min, max components.
        :rtype: tuple[Any, Any, Any]
        """
        if mode == "free":
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        if mode == "fixed":
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        # constrained
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    return handler


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

    # Set initial visibility based on mode
    show_constraint = mode == "constrained"
    show_min_max = mode == "free"

    with gr.Row():
        with gr.Column(scale=3, min_width=200):
            gr.Markdown(label)
        value_comp = gr.Number(
            value=fv,
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
            value=str(constraint) if constraint else "",
            label="Constraint expression",
            info='Expression evaluated after each step, e.g. "16.5 / sin(radians(system@inclination))".',
            scale=6,
            visible=show_constraint,
            interactive=True,
            container=True,
        )
        min_comp = gr.Number(
            value=flo,
            label="Min",
            scale=2,
            visible=show_min_max,
            interactive=True,
            container=True,
        )
        max_comp = gr.Number(
            value=fhi,
            label="Max",
            scale=2,
            visible=show_min_max,
            interactive=True,
            container=True,
        )

    # Wire the mode change handler BEFORE storing in dict
    mode_comp.change(
        fn=_make_mode_change_handler(),
        inputs=[mode_comp],
        outputs=[constraint_comp, min_comp, max_comp],
    )

    # Now store in components dict
    components[f"{section}_{name}_value"] = value_comp
    components[f"{section}_{name}_mode"] = mode_comp
    components[f"{section}_{name}_constraint"] = constraint_comp
    components[f"{section}_{name}_min"] = min_comp
    components[f"{section}_{name}_max"] = max_comp


def _build_component_section(
    components: dict[str, gr.Component],
    section: str,
    approach: Literal["community", "standard"],
    defaults: dict[str, Float | bool | str | None],
    value_overrides: dict[str, tuple[float, float, float]] | None = None,
) -> None:
    """Render one primary/secondary component section.

    :param components: Mutable dict that receives the new components.
    :type components: dict[str, gr.Component]
    :param section: Section prefix (``"primary"`` or ``"secondary"``).
    :type section: str
    :param approach: Fitting approach (``"community"`` or ``"standard"``).
    :type approach: Literal["community", "standard"]
    :param defaults: User-supplied defaults override dict.
    :type defaults: dict[str, Float | bool | str | None]
    :param value_overrides: Optional per-name ``(value, min, max)`` overrides used
        for the secondary component whose default values differ from the primary.
    :type value_overrides: dict[str, tuple[float, float, float]] | None
    """
    # Build combined spec based on approach
    if approach == "standard":
        combined_spec = {**_COMPONENT_STANDARD_SPEC, **_COMPONENT_COMMON_SPEC}
    else:
        combined_spec = _COMPONENT_COMMON_SPEC

    for name, (label, def_val, def_fixed, def_min, def_max, _unit) in combined_spec.items():
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

    Creates two tabs (Community and Standard) with different parameter sets.
    Each approach has three labelled sections (System, Primary, Secondary)
    plus a nuisance row for MCMC.  An optional *defaults* dict overrides the
    built-in starting values; unrecognised keys are silently ignored.

    :param defaults: Flat mapping of ``"{section}_{name}_{sub}"`` keys
        to override default values.
    :type defaults: dict[str, Float | bool | str | None] | None
    :returns: Dict keyed by field names from both approaches.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    components: dict[str, gr.Component] = {}


    with gr.Tabs():
        # ============================================================== #
        # Community Approach Tab                                          #
        # ============================================================== #
        with gr.Tab("Community"):
            _build_approach_params(components, "community", defaults)

        # ============================================================== #
        # Standard Approach Tab                                           #
        # ============================================================== #
        with gr.Tab("Standard"):
            _build_approach_params(components, "standard", defaults)

    return components


def _build_approach_params(
    components: dict[str, gr.Component],
    approach: Literal["community", "standard"],
    defaults: dict[str, Float | bool | str | None],
) -> None:
    """Build parameter inputs for a specific approach.

    :param components: Mutable dict that receives the new components.
    :type components: dict[str, gr.Component]
    :param approach: Fitting approach (``"community"`` or ``"standard"``).
    :type approach: Literal["community", "standard"]
    :param defaults: User-supplied defaults override dict.
    :type defaults: dict[str, Float | bool | str | None]
    """
    # Determine which parameter sets to use
    system_params = SYSTEM_PARAMS_COMMUNITY if approach == "community" else SYSTEM_PARAMS_STANDARD

    # Build combined system spec
    system_spec = {**_SYSTEM_COMMON_SPEC, **_SYSTEM_COMMUNITY_SPEC} if approach == "community" else _SYSTEM_COMMON_SPEC

    # ------------------------------------------------------------------ #
    # Section: System                                                       #
    # ------------------------------------------------------------------ #
    with gr.Accordion("System Parameters", open=True):
        for name in system_params:
            label, def_val, def_fixed, def_min, def_max, _unit = system_spec[name]
            val = defaults.get(f"system_{name}_value", def_val)
            fixed = defaults.get(f"system_{name}_fixed", def_fixed)
            lo = defaults.get(f"system_{name}_min", def_min)
            hi = defaults.get(f"system_{name}_max", def_max)
            constraint = defaults.get(f"system_{name}_constraint")
            _param_row(components, "system", name, label, val, fixed, lo, hi, constraint)

    # ------------------------------------------------------------------ #
    # Section: Primary                                                      #
    # ------------------------------------------------------------------ #
    with gr.Accordion("Primary Component Parameters", open=False):
        _build_component_section(components, "primary", approach, defaults)

    # ------------------------------------------------------------------ #
    # Section: Secondary                                                    #
    # ------------------------------------------------------------------ #
    with gr.Accordion("Secondary Component Parameters", open=False):
        _secondary_value_overrides: dict[str, tuple[float, float, float]] = {
            "mass": (1.0, 0.1, 100.0),
            "t_eff": (4000.0, 4000.0, 7000.0),
            "surface_potential": (5.0, 5.0, 7.0),
            "gravity_darkening": (0.32, 0.0, 1.0),
            "albedo": (0.6, 0.0, 1.0),
            "synchronicity": (1.0, 0.1, 10.0),
            "metallicity": (0.0, -2.0, 1.0),
        }
        _build_component_section(
            components, "secondary", approach, defaults,
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


