"""Gradio component builder for pulsation-mode inputs in the LC Modeling tab.

Each star can optionally have one or more pulsation modes. Modes are added via
an "Add pulsation mode" button; each rendered panel has its own "Remove" button
so any specific mode can be deleted. Removing shifts subsequent modes up so the
active list is always contiguous from the top.

Default input units match :data:`elisa.units.DefaultPulsationsInputUnits`:

- **amplitude** - m/s
- **frequency** - cycles/day (1/d)
- **start_phase** - radians
- **mode_axis_theta** / **mode_axis_phi** - degrees
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import gradio as gr

from elisa.ui.shared import build_full_width_button_row

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MAX_MODES: int = 5

_MODE_PARAMS: tuple[str, ...] = (
    "l",
    "m",
    "amplitude",
    "frequency",
    "start_phase",
    "mode_axis_theta",
    "mode_axis_phi",
    "temperature_perturbation_phase_shift",
    "horizontal_to_radial_amplitude_ratio",
    "temperature_amplitude_factor",
    "tidally_locked",
)

#: Canonical key order for the flat Gradio value list used by the tab handler.
FIELD_ORDER: tuple[str, ...] = (
    "enabled",
    "mode_count",
    *(f"mode_{i}_{p}" for i in range(MAX_MODES) for p in _MODE_PARAMS),
)

_DEFAULTS: dict[str, object] = {
    "l": 6,
    "m": 3,
    "amplitude": 500.0,
    "frequency": 15.0,
    # leave empty by default so user must explicitly set if desired
    "start_phase": "",
    "mode_axis_theta": 25.0,
    "mode_axis_phi": 90.0,
    # leave temperature perturbation phase shift empty by default so user must optional
    "temperature_perturbation_phase_shift": "",
    # leave H-to-R empty by default so user can optional explicitly
    "horizontal_to_radial_amplitude_ratio": "",
    "temperature_amplitude_factor": 0.01,
    "tidally_locked": False,
}


# ---------------------------------------------------------------------------
# Internal handler factories
# ---------------------------------------------------------------------------


def _make_add_handler(max_modes: int) -> Callable[[int], list[object]]:
    """Return a handler that increments the mode count and shows the next slot.

    :param max_modes: Maximum number of allowed modes.
    :type max_modes: int
    :returns: Callable suitable for ``gr.Button.click``.
    :rtype: Callable[[int], list[object]]
    """

    def handler(mode_count: int) -> list[object]:
        current = int(mode_count)
        new_count = min(current + 1, max_modes)
        return [new_count, *[gr.update(visible=(i < new_count)) for i in range(max_modes)]]

    return handler


def _make_remove_handler(slot_idx: int, max_modes: int) -> Callable[..., list[object]]:
    """Return a handler that removes one mode slot and shifts later slots up.

    :param slot_idx: Zero-based index of the mode to remove.
    :type slot_idx: int
    :param max_modes: Total number of pre-built mode slots.
    :type max_modes: int
    :returns: Callable suitable for ``gr.Button.click``.
    :rtype: Callable[..., list[object]]
    """
    n = len(_MODE_PARAMS)
    default_row: list[object] = [_DEFAULTS[p] for p in _MODE_PARAMS]

    def handler(mode_count: int, *flat_values: object) -> list[object]:
        current = int(mode_count)
        total_outputs = 1 + max_modes + max_modes * n
        if current == 0:
            return [gr.update()] * total_outputs

        new_count = max(0, current - 1)

        # Rebuild per-slot value lists from the flat sequence
        modes = [list(flat_values[k * n : (k + 1) * n]) for k in range(max_modes)]

        # Shift everything above slot_idx down by one
        for k in range(slot_idx, max_modes - 1):
            modes[k] = modes[k + 1]
        # Reset the last slot to defaults
        modes[-1] = list(default_row)

        out: list[object] = [new_count]
        out.extend(gr.update(visible=(k < new_count)) for k in range(max_modes))
        for k in range(max_modes):
            out.extend(gr.update(value=modes[k][j]) for j in range(n))
        return out

    return handler


# ---------------------------------------------------------------------------
# Public API - component builder
# ---------------------------------------------------------------------------


def build(prefix: str) -> dict[str, gr.Component]:
    """Render an optional pulsation section for one stellar component.

    :param prefix: Human-readable label prefix, e.g. ``"Primary"`` or
        ``"Secondary"``, used in the accordion title and checkbox label.
    :type prefix: str
    :returns: Ordered dict mapping all :data:`FIELD_ORDER` keys to components.
    :rtype: dict[str, gr.Component]
    """
    components: dict[str, gr.Component] = {}
    value_comps: dict[str, gr.Component] = {}

    with gr.Accordion(f"{prefix} Pulsations", open=False):
        enabled_comp = gr.Checkbox(
            label=f"Enable {prefix.lower()} pulsations",
            value=False,
            info="Tick to add pulsation modes to this component.",
        )
        mode_count = gr.State(value=0)

        with gr.Column(visible=False) as puls_section:
            # Global add button for all modes - visually separated from the first mode panel.
            add_btn = build_full_width_button_row(
                "+ Add pulsation mode",
                elem_classes=["full-width-button"],
                spacer_margin_px=8,
            )

            mode_groups: list[gr.Column] = []
            remove_btns: list[gr.Button] = []

            for puls_mode_idx in range(MAX_MODES):
                with gr.Column(visible=False) as grp:
                    if puls_mode_idx > 0:
                        gr.HTML("<hr style='margin: 10px 0;'>")

                    # Centered mode label for better visual balance.
                    gr.Markdown(
                        f"<div style='text-align: center; font-weight: 600;'>Mode {puls_mode_idx + 1}</div>",
                        elem_classes=["pulsation-mode-header"],
                    )

                    # Full-width remove button under the header.
                    remove_btn_i = gr.Button(
                        "✕ Remove pulsation mode",
                        variant="stop",
                        size="sm",
                        scale=1,
                    )

                    with gr.Row():
                        value_comps[f"mode_{puls_mode_idx}_l"] = gr.Number(
                            label="l (degree)",
                            value=int(_DEFAULTS["l"]),  # type: ignore[arg-type]
                            precision=0,
                            info="Spherical harmonic degree (≥ 0, integer)",
                        )
                        value_comps[f"mode_{puls_mode_idx}_m"] = gr.Number(
                            label="m (order)",
                            value=int(_DEFAULTS["m"]),  # type: ignore[arg-type]
                            precision=0,
                            info="Azimuthal order (integer, |m| ≤ l)",
                        )

                    with gr.Row():
                        value_comps[f"mode_{puls_mode_idx}_amplitude"] = gr.Number(
                            label="Amplitude (m/s)",
                            value=float(_DEFAULTS["amplitude"]),  # type: ignore[arg-type]
                            info="Radial velocity amplitude in m/s.",
                        )
                        value_comps[f"mode_{puls_mode_idx}_frequency"] = gr.Number(
                            label="Frequency (1/d)",
                            value=float(_DEFAULTS["frequency"]),  # type: ignore[arg-type]
                            info="Pulsation frequency in cycles per day.",
                        )

                    with gr.Row():
                        value_comps[f"mode_{puls_mode_idx}_start_phase"] = gr.Number(
                            label="Start phase (rad)",
                            value=(
                                None
                                if cast("str", _DEFAULTS["start_phase"]) == ""
                                else float(cast("float", _DEFAULTS["start_phase"]))
                            ),
                            info="Initial phase offset in radians.",
                        )
                        value_comps[f"mode_{puls_mode_idx}_mode_axis_theta"] = gr.Number(
                            label="Mode axis θ (deg)",
                            value=float(_DEFAULTS["mode_axis_theta"]),  # type: ignore[arg-type]
                            info="Latitude of mode axis in degrees.",
                        )

                    with gr.Row():
                        value_comps[f"mode_{puls_mode_idx}_mode_axis_phi"] = gr.Number(
                            label="Mode axis φ (deg)",
                            value=float(_DEFAULTS["mode_axis_phi"]),  # type: ignore[arg-type]
                            info="Longitude of mode axis in degrees.",
                        )

                    with gr.Group():
                        gr.Markdown("**Optional Parameters**")

                    # Each optional parameter on a separate row so empty vs 0 is visually and
                    # functionally distinct and the layout is easier to read.
                    with gr.Row():
                        # Use Textbox so empty string is preserved - user may want to leave unset
                        value_comps[
                            f"mode_{puls_mode_idx}_temperature_perturbation_phase_shift"
                        ] = gr.Textbox(
                            label="Temp. phase shift (rad)",
                            value=(
                                ""
                                if cast("str", _DEFAULTS["temperature_perturbation_phase_shift"]) == ""
                                else str(cast("float", _DEFAULTS["temperature_perturbation_phase_shift"]))
                            ),
                            placeholder="leave empty to use default",
                            lines=1,
                            info="Phase lag between temperature and radial displacement.",
                        )

                    with gr.Row():
                        value_comps[
                            f"mode_{puls_mode_idx}_horizontal_to_radial_amplitude_ratio"
                        ] = gr.Textbox(
                            label="H-to-R amplitude ratio",
                            value=(
                                ""
                                if cast("str", _DEFAULTS["horizontal_to_radial_amplitude_ratio"]) == ""
                                else str(cast("float", _DEFAULTS["horizontal_to_radial_amplitude_ratio"]))
                            ),
                            placeholder="leave empty to use default",
                            lines=1,
                            info="Ratio of horizontal to radial amplitude.",
                        )

                    with gr.Row():
                        value_comps[f"mode_{puls_mode_idx}_temperature_amplitude_factor"] = gr.Number(
                            label="Temp. amplitude factor",
                            value=float(cast("float", _DEFAULTS["temperature_amplitude_factor"])),
                            info="Ratio ΔT/T_eff (temperature amplitude).",
                        )

                    with gr.Row():
                        value_comps[f"mode_{puls_mode_idx}_tidally_locked"] = gr.Checkbox(
                            label="Tidally locked",
                            value=bool(_DEFAULTS["tidally_locked"]),
                            info="Hold mode axis position relative to companion.",
                        )

                mode_groups.append(grp)
                remove_btns.append(remove_btn_i)

        # Event wiring
        enabled_comp.change(
            fn=lambda v: gr.update(visible=v),
            inputs=[enabled_comp],
            outputs=[puls_section],
        )

        add_btn.click(
            fn=_make_add_handler(MAX_MODES),
            inputs=[mode_count],
            outputs=[mode_count, *mode_groups],
        )

        all_value_comps: list[gr.Component] = [
            value_comps[f"mode_{puls_mode_idx}_{p}"] for puls_mode_idx in range(MAX_MODES) for p in _MODE_PARAMS
        ]

        for puls_mode_remove_idx, rm_btn in enumerate(remove_btns):
            rm_btn.click(
                fn=_make_remove_handler(puls_mode_remove_idx, MAX_MODES),
                inputs=[mode_count, *all_value_comps],
                outputs=[mode_count, *mode_groups, *all_value_comps],
            )

    components["enabled"] = enabled_comp
    components["mode_count"] = mode_count
    components.update(value_comps)
    return components


# ---------------------------------------------------------------------------
# Public API - parsing
# ---------------------------------------------------------------------------


def parse_pulsation_modes(pulsation_params: dict[str, object] | None) -> list[dict[str, object]]:
    """Convert flat pulsation params to a list of mode parameter dicts.

    :param pulsation_params: Flat dict keyed by :data:`FIELD_ORDER` as
        produced by :func:`build`. Pass ``None`` to disable pulsations.
    :type pulsation_params: dict[str, object] | None
    :returns: List of mode parameter dicts suitable for ``Star(pulsations=[...])``.
    :rtype: list[dict[str, object]]
    """
    if not pulsation_params or not bool(pulsation_params.get("enabled", False)):
        return []

    mode_count = int(pulsation_params.get("mode_count", 0) or 0)
    if mode_count <= 0:
        return []

    modes: list[dict[str, object]] = []
    for puls_mode_parse_idx in range(min(mode_count, MAX_MODES)):
        # Get mandatory parameters with defaults
        l_val: int | None = pulsation_params.get(f"mode_{puls_mode_parse_idx}_l")
        l_degree: int = l_val if l_val is not None else int(cast("int", _DEFAULTS["l"]))

        m_val: int | None = pulsation_params.get(f"mode_{puls_mode_parse_idx}_m")
        m_order: int = m_val if m_val is not None else int(cast("int", _DEFAULTS["m"]))

        amplitude_val: float | None = pulsation_params.get(f"mode_{puls_mode_parse_idx}_amplitude")
        amplitude: float = amplitude_val if amplitude_val is not None else float(cast("float", _DEFAULTS["amplitude"]))

        frequency_val: float | None = pulsation_params.get(f"mode_{puls_mode_parse_idx}_frequency")
        frequency: float = frequency_val if frequency_val is not None else float(cast("float", _DEFAULTS["frequency"]))

        mode: dict[str, object] = {
            "l": l_degree,
            "m": m_order,
            "amplitude": amplitude,
            "frequency": frequency,
        }

        # Handle optional parameters - skip if None or empty
        for param in (
            "start_phase",
            "mode_axis_theta",
            "mode_axis_phi",
            "temperature_perturbation_phase_shift",
            "horizontal_to_radial_amplitude_ratio",
            "temperature_amplitude_factor",
        ):
            raw = pulsation_params.get(f"mode_{puls_mode_parse_idx}_{param}")
            # temperature_perturbation_phase_shift and horizontal_to_radial_amplitude_ratio
            # are rendered as Textbox to preserve empty vs '0'. Accept strings and numbers.
            if raw is None:
                continue
            if isinstance(raw, str):
                if raw.strip() == "":
                    continue
                try:
                    parsed = float(raw.strip())
                except ValueError:
                    # Non-numeric input - skip and let higher-level validation handle it
                    continue
            else:
                # numeric input (int/float)
                parsed = float(cast("float | int", raw))

            mode[param] = parsed

        # Handle boolean parameter separately - only include if explicitly set
        tidally_locked_val: bool | None = pulsation_params.get(f"mode_{puls_mode_parse_idx}_tidally_locked")
        if tidally_locked_val is not None:
            mode["tidally_locked"] = bool(tidally_locked_val)

        modes.append(mode)

    return modes
