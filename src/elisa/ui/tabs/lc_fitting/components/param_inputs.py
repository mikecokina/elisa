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
SYSTEM_COMMON_SPEC: dict[str, _Spec] = {
    "inclination": (
        "**Inclination**  i  [deg]",
        85.0, False, 80.0, 90.0, "deg",
    ),
    "eccentricity": (
        "**Eccentricity**  e",
        0.0, True, 0.0, 0.5, None,
    ),
    "argument_of_periastron": (
        "**Arg. of periastron**  ω  [deg]",
        0.0, True, 0.0, 360.0, "deg",
    ),
    "period": (
        "**Orbital period**  P  [d]",
        2.5, True, 0.01, 1000.0, "d",
    ),
    "primary_minimum_time": (
        "**Primary minimum time**  T₀  [d]",
        54953.5388437, True, 50000.0, 60000.0, "d",
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
        11.55, False, 5.0, 30.0, "R☉",
    ),
    "mass_ratio": (
        "**Mass ratio**  q = M₂/M₁",
        0.56, True, 0.1, 2.0, None,
    ),
}

# Component parameters common to both approaches
_COMPONENT_COMMON_SPEC: dict[str, _Spec] = {
    "t_eff": (
        "**Effective temperature**  T_eff  [K]",
        9500.0, False, 9000.0, 11000.0, "K",
    ),
    "surface_potential": (
        "**Surface potential**  Ω",
        4.0, False, 3.0, 5.0, None,
    ),
    "gravity_darkening": (
        "**Gravity darkening**  β",
        1.0, True, 0.0, 1.0, None,
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

MAX_SPOT_SLOTS = 6
MAX_SPOTS_LOCAL = 6

SPOT_PARAMS: tuple[str, ...] = (
    "longitude",
    "latitude",
    "angular_radius",
    "temperature_factor",
)

# sensible defaults for new spot rows
_SPOT_DEFAULTS: dict[str, tuple[float, float, float]] = {
    "longitude": (230.0, 180.0, 270.0),
    "latitude": (45.0, 0.0, 90.0),
    "angular_radius": (50.0, 45.0, 80.0),
    "temperature_factor": (0.98, 0.93, 1.0),
}

# Unit map for spot parameters
SPOT_UNITS: dict[str, str] = {
    "longitude": "deg",
    "latitude": "deg",
    "angular_radius": "deg",
    "temperature_factor": None,  # dimensionless
}

# ---------------------------------------------------------------------------
# Exported name tuples
# ---------------------------------------------------------------------------

#: System params for community approach (common + semi_major_axis + mass_ratio).
SYSTEM_PARAMS_COMMUNITY: tuple[str, ...] = (
    *tuple(SYSTEM_COMMON_SPEC.keys()),
    *tuple(_SYSTEM_COMMUNITY_SPEC.keys()),
)

#: System params for standard approach (common only, no semi_major_axis or mass_ratio).
SYSTEM_PARAMS_STANDARD: tuple[str, ...] = tuple(SYSTEM_COMMON_SPEC.keys())

#: Component params for community approach (no mass).
COMPONENT_PARAMS_COMMUNITY: tuple[str, ...] = tuple(_COMPONENT_COMMON_SPEC.keys())

#: Component params for standard approach (mass + common).
COMPONENT_PARAMS_STANDARD: tuple[str, ...] = (
    *tuple(_COMPONENT_STANDARD_SPEC.keys()),
    *tuple(_COMPONENT_COMMON_SPEC.keys()),
)

# Backward compatibility exports - default to community approach
SYSTEM_PARAMS: tuple[str, ...] = SYSTEM_PARAMS_COMMUNITY
SYSTEM_REGULAR_PARAMS: tuple[str, ...] = tuple(SYSTEM_COMMON_SPEC.keys())
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
        # Spots for primary - enabled, count, then per-spot fields
        "primary_spots_enabled",
        "primary_spot_count",
        *(
            f"primary_spot_{i}_{p}_{s}"
            for i in range(6)
            for p in ("longitude", "latitude", "angular_radius", "temperature_factor")
            for s in ("value", "mode", "constraint", "min", "max")
        ),
        *(
            f"secondary_{name}_{sub}"
            for name in component_params
            for sub in ("value", "mode", "constraint", "min", "max")
        ),
        # Spots for secondary - enabled, count, then per-spot fields
        "secondary_spots_enabled",
        "secondary_spot_count",
        *(
            f"secondary_spot_{i}_{p}_{s}"
            for i in range(6)
            for p in ("longitude", "latitude", "angular_radius", "temperature_factor")
            for s in ("value", "mode", "constraint", "min", "max")
        ),
        "nuisance_ln_f_value",
        "nuisance_ln_f_mode",
        "nuisance_ln_f_constraint",
        "nuisance_ln_f_min",
        "nuisance_ln_f_max",
    )


# Default to community approach for backward compatibility
FIELD_ORDER: tuple[str, ...] = _build_field_order("community")
FIELD_ORDER_COMMUNITY: tuple[str, ...] = _build_field_order("community")
FIELD_ORDER_STANDARD: tuple[str, ...] = _build_field_order("standard")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mode_handler(mode: str) -> tuple[Any, Any, Any]:
    """Handle mode radio change for any parameter row.

    Controls which secondary fields are editable based on the selected mode:

    - **free** - min and max are editable, constraint is greyed out.
    - **fixed** - min, max, and constraint are all greyed out.
    - **constrained** - constraint is editable, min and max are greyed out.

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
    section: str,
    name: str,
    label: str,
    value: float | str | None ,
    fixed: bool,  # noqa: FBT001
    lo: float | str | None,
    hi: float | str | None,
    constraint: str | None = None,
) -> None:
    """Render one parameter row with mode selector (free/fixed/constrained).

    All secondary fields (constraint, min, max) are always visible.
    Interactivity is toggled by the mode radio so Gradio never needs to
    insert or remove DOM elements, which avoids the loading-spinner bug
    that affects ``gr.update(visible=...)`` inside nested Tabs/Accordions.

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
    if constraint:
        mode = "constrained"
    elif bool(fixed):
        mode = "fixed"
    else:
        mode = "free"

    fv: float | None = float(value) if value is not None else None  # type: ignore[arg-type]
    flo: float | None = float(lo) if lo is not None else None  # type: ignore[arg-type]
    fhi: float | None = float(hi) if hi is not None else None  # type: ignore[arg-type]

    constraint_active = mode == "constrained"
    bounds_active = mode == "free"

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
            placeholder='e.g. "11.2 / sin(radians(system@inclination))"',
            scale=4,
            interactive=constraint_active,
            container=True,
        )
        min_comp = gr.Number(
            value=flo,
            label="Min",
            scale=2,
            interactive=bounds_active,
            container=True,
        )
        max_comp = gr.Number(
            value=fhi,
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

    # ------------------------------------------------------------------ #
    # Spots UI for this component                                          #
    # ------------------------------------------------------------------ #
    from elisa.ui.shared import build_full_width_button_row  # noqa: PLC0415

    def _make_spot_add_handler(max_spots: int):
        def handler(spot_count: int):
            current = int(spot_count)
            new_count = min(current + 1, max_spots)
            return [new_count, *[gr.update(visible=(i < new_count)) for i in range(max_spots)]]

        return handler

    def _make_spot_remove_handler(slot_idx: int, max_spots: int) -> Callable:
        n_params = len(SPOT_PARAMS)
        n_sub = 5  # value, mode, constraint, min, max
        default_row = []
        for p in SPOT_PARAMS:
            dv, dlo, dhi = _SPOT_DEFAULTS[p]
            default_row.extend([dv, "free", "", dlo, dhi])

        def handler(spot_count_: int, *flat_values: object) -> list[object]:
            current = int(spot_count_)
            total_outputs = 1 + max_spots + max_spots * n_params * n_sub
            if current == 0:
                return [gr.update()] * total_outputs

            new_count = max(0, current - 1)

            # reconstruct spots array from flat_values
            slots: list[list[object]] = []
            idx = 0
            for k in range(max_spots):
                slot_vals: list[object] = []
                for _ in range(n_params * n_sub):
                    slot_vals.append(flat_values[idx])
                    idx += 1
                slots.append(slot_vals)

            # shift left from slot_idx
            for k in range(slot_idx, max_spots - 1):
                slots[k] = slots[k + 1]
            slots[-1] = list(default_row)

            out: list[object] = [new_count]
            out.extend(gr.update(visible=(k < new_count)) for k in range(max_spots))
            for k in range(max_spots):
                for val_ in slots[k]:
                    out.extend([gr.update(value=val_)])
            return out

        return handler

    with gr.Accordion("Spots (optional)", open=False):
        enabled_comp = gr.Checkbox(
            label=f"Enable {section.capitalize()} spots",
            value=False,
            info="Tick to add surface spots to this component.",
        )
        spot_count = gr.State(value=0)

        with gr.Column(visible=False) as spots_section:
            add_btn = build_full_width_button_row(
                "+ Add spot",
                elem_classes=["full-width-button"],
                spacer_margin_px=8,
            )

            spot_groups: list[gr.Column] = []
            remove_btns: list[gr.Button] = []

            for i in range(MAX_SPOTS_LOCAL):
                with gr.Column(visible=False) as grp:
                    if i > 0:
                        gr.HTML("<hr style='margin: 10px 0;'>")

                    gr.Markdown(f"**Spot {i + 1}**", elem_classes=["spot-header"])

                    remove_btn = gr.Button(
                        "✕ Remove spot",
                        variant="stop",
                        size="sm",
                        scale=1,
                    )

                    # per-spot parameters - use _param_row to create full control set
                    for p in SPOT_PARAMS:
                        lab = {
                            "longitude": "Longitude (deg)",
                            "latitude": "Latitude (deg)",
                            "angular_radius": "Angular radius (deg)",
                            "temperature_factor": "Temperature factor",
                        }[p]
                        # build keys as e.g. primary_spot_0_longitude_value
                        default_val = _SPOT_DEFAULTS[p][0]
                        default_lo = _SPOT_DEFAULTS[p][1]
                        default_hi = _SPOT_DEFAULTS[p][2]
                        _param_row(
                            components,
                            section,
                            f"spot_{i}_{p}",
                            f"{lab}",
                            defaults.get(f"{section}_spot_{i}_{p}_value", default_val),
                            defaults.get(f"{section}_spot_{i}_{p}_fixed", False),
                            defaults.get(f"{section}_spot_{i}_{p}_min", default_lo),
                            defaults.get(f"{section}_spot_{i}_{p}_max", default_hi),
                            defaults.get(f"{section}_spot_{i}_{p}_constraint"),
                        )

                spot_groups.append(grp)
                remove_btns.append(remove_btn)

        enabled_comp.change(
            fn=lambda v: gr.update(visible=v),
            inputs=[enabled_comp],
            outputs=[spots_section],
        )

        add_btn.click(
            fn=_make_spot_add_handler(MAX_SPOTS_LOCAL),
            inputs=[spot_count],
            outputs=[spot_count, *spot_groups],
        )

        # prepare list of all flat value components for remove handlers
        all_spot_value_comps: list[gr.Component] = []
        for i in range(MAX_SPOTS_LOCAL):
            for p in SPOT_PARAMS:
                for sub in ("value", "mode", "constraint", "min", "max"):
                    all_spot_value_comps.append(components[f"{section}_spot_{i}_{p}_{sub}"])

        for i, rm_btn in enumerate(remove_btns):
            rm_btn.click(
                fn=_make_spot_remove_handler(i, MAX_SPOTS_LOCAL),
                inputs=[spot_count, *all_spot_value_comps],
                outputs=[spot_count, *spot_groups, *all_spot_value_comps],
            )

        # register components into mapping so FIELD_ORDER keys exist
        components[f"{section}_spots_enabled"] = enabled_comp
        components[f"{section}_spot_count"] = spot_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build(
    *,
    approach: str = "community",
    defaults: dict[str, Float | bool | str | None] | None = None,
) -> tuple[dict[str, gr.Component], list[gr.Component]]:
    """Render initial parameters for a specific approach.

    Builds ONLY the specified approach (Community or Standard). Returns its own
    field order tuple and the parameter components dict.

    :param approach: Fitting approach (``"community"`` or ``"standard"``).
    :type approach: str
    :param defaults: Flat mapping of ``"{section}_{name}_{sub}"`` keys
        to override default values.
    :type defaults: dict[str, Float | bool | str | None] | None
    :returns: Tuple of ``(components_dict, sections_list)``.
    :rtype: tuple[dict[str, gr.Component], list[gr.Component]]
    """
    if defaults is None:
        defaults = {}

    components: dict[str, gr.Component] = {}
    sections: list[gr.Component] = []

    # noinspection PyTypeChecker
    _build_approach_params(components, approach, defaults)

    return components, sections


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
    system_spec = {**SYSTEM_COMMON_SPEC, **_SYSTEM_COMMUNITY_SPEC} if approach == "community" else SYSTEM_COMMON_SPEC

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
            # semi_major_axis defaults to constrained mode in community approach
            if name == "semi_major_axis" and approach == "community":
                constraint = defaults.get(
                    f"system_{name}_constraint",
                    "11.55 / sin(radians(system@inclination))",
                )
            else:
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
            "t_eff": (7850.0, 7000.0, 8000.0),
            "surface_potential": (4.5, 3.2, 5.5),
            "gravity_darkening": (1.0, 0.0, 1.0),
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


def parse_spots_fit(spot_params: dict[str, object] | None, section: str | None = None) -> list[dict[str, object]]:
    """Convert flat spot parameters (from the fitting form) into a list of spot dicts.

    Each spot parameter in the form has controls for value/mode/constraint/min/max
    and this helper extracts the values and constructs the list suitable for
    serialization in the ELISa fit JSON, matching the SpotInitialParameters format:

    ``[{"label": "spot1", "longitude": {"value": ..., "fixed": ..., "min": ..., "max": ..., "unit": ...}, ...}, ...]``

    :param spot_params: Flat dict produced by the fitting components build.
    :type spot_params: dict[str, object] | None
    :param section: Optional section prefix (``"primary"`` / ``"secondary"``) used
        when keys are namespaced; if provided the function will look for
        ``"{section}_spot_{i}_{p}_value"`` keys.
    :type section: str | None
    :returns: List of spot dicts with label and parameter structure.
    :rtype: list[dict[str, object]]
    """
    from typing import cast

    if not spot_params:
        return []

    enabled_key = f"{section}_spots_enabled" if section else "spots_enabled"
    count_key = f"{section}_spot_count" if section else "spot_count"
    if not bool(spot_params.get(enabled_key, False)):
        return []

    count = int(spot_params.get(count_key, 0) or 0)
    if count <= 0:
        return []

    spots: list[dict[str, object]] = []

    for i in range(min(count, MAX_SPOT_SLOTS)):
        def get_key(p: str, sub: str) -> str:
            base = f"{section}_spot_{i}_{p}" if section else f"spot_{i}_{p}"
            return f"{base}_{sub}"

        # Build the spot dict with label and parameter entries
        spot: dict[str, object] = {
            "label": f"spot{i + 1}",
        }

        # For each spot parameter (longitude, latitude, angular_radius, temperature_factor)
        for param_name in ("longitude", "latitude", "angular_radius", "temperature_factor"):
            value = spot_params.get(get_key(param_name, "value"))
            mode = str(spot_params.get(get_key(param_name, "mode"), "free"))
            constraint = spot_params.get(get_key(param_name, "constraint"))
            min_val = spot_params.get(get_key(param_name, "min"))
            max_val = spot_params.get(get_key(param_name, "max"))

            # Convert value to float
            fvalue = (
                float(cast("float | int", value))
                if value is not None and str(value).strip() != ""
                else {
                    "longitude": 230.0,
                    "latitude": 45.0,
                    "angular_radius": 50.0,
                    "temperature_factor": 0.98,
                }[param_name]
            )

            # Build parameter entry
            param_entry: dict[str, object] = {"value": fvalue}

            # Handle mode
            if mode == "constrained":
                param_entry["constraint"] = str(constraint or "")
            elif mode == "fixed":
                param_entry["fixed"] = True
            else:  # free
                param_entry["fixed"] = False
                # Only include min/max for free mode
                if min_val is not None and str(min_val).strip() != "":
                    param_entry["min"] = float(cast("float | int", min_val))
                if max_val is not None and str(max_val).strip() != "":
                    param_entry["max"] = float(cast("float | int", max_val))

            # Add unit if applicable
            unit = SPOT_UNITS.get(param_name)
            if unit is not None:
                param_entry["unit"] = unit

            spot[param_name] = param_entry

        spots.append(spot)

    return spots
