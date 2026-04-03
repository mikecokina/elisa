"""Gradio component builder for pulsation-mode inputs in the LC Modeling tab.

Each star can optionally have several pulsation modes. All mode slots are
always rendered and each slot has its own enable checkbox. Disabled slots are
shown but non-interactive (greyed out). Only enabled slots are serialized by
``parse_pulsation_modes``.

Default input units match :data:`elisa.units.DefaultPulsationsInputUnits`:

- **amplitude** - m/s
- **frequency** - cycles/day (1/d)
- **start_phase** - radians
- **mode_axis_theta** / **mode_axis_phi** - degrees
"""

from __future__ import annotations

from typing import cast

import gradio as gr

from elisa.ui.shared.const import MAX_PULSE_MODES

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

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
    *(f"mode_{i}_enabled" for i in range(MAX_PULSE_MODES)),
    *(f"mode_{i}_{p}" for i in range(MAX_PULSE_MODES) for p in _MODE_PARAMS),
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
        for puls_mode_idx in range(MAX_PULSE_MODES):
            if puls_mode_idx > 0:
                gr.HTML("<hr style='margin: 10px 0;'>")

            with gr.Row():
                gr.Markdown(f"**Mode {puls_mode_idx + 1}**", elem_classes=["pulsation-mode-header"])
                enabled_comp = gr.Checkbox(
                    label="Use",
                    value=False,
                    scale=0,
                    min_width=80,
                )

            with gr.Row():
                l_comp = gr.Number(
                    label="l (degree)",
                    value=int(_DEFAULTS["l"]),  # type: ignore[arg-type]
                    precision=0,
                    info="Spherical harmonic degree (>= 0, integer)",
                    interactive=False,
                )
                m_comp = gr.Number(
                    label="m (order)",
                    value=int(_DEFAULTS["m"]),  # type: ignore[arg-type]
                    precision=0,
                    info="Azimuthal order (integer, |m| <= l)",
                    interactive=False,
                )

            with gr.Row():
                amplitude_comp = gr.Number(
                    label="Amplitude (m/s)",
                    value=float(_DEFAULTS["amplitude"]),  # type: ignore[arg-type]
                    info="Radial velocity amplitude in m/s.",
                    interactive=False,
                )
                frequency_comp = gr.Number(
                    label="Frequency (1/d)",
                    value=float(_DEFAULTS["frequency"]),  # type: ignore[arg-type]
                    info="Pulsation frequency in cycles per day.",
                    interactive=False,
                )

            with gr.Row():
                start_phase_comp = gr.Number(
                    label="Start phase (rad)",
                    value=(
                        None
                        if cast("str", _DEFAULTS["start_phase"]) == ""
                        else float(cast("float", _DEFAULTS["start_phase"]))
                    ),
                    info="Initial phase offset in radians.",
                    interactive=False,
                )
                mode_axis_theta_comp = gr.Number(
                    label="Mode axis theta (deg)",
                    value=float(_DEFAULTS["mode_axis_theta"]),  # type: ignore[arg-type]
                    info="Latitude of mode axis in degrees.",
                    interactive=False,
                )

            with gr.Row():
                mode_axis_phi_comp = gr.Number(
                    label="Mode axis phi (deg)",
                    value=float(_DEFAULTS["mode_axis_phi"]),  # type: ignore[arg-type]
                    info="Longitude of mode axis in degrees.",
                    interactive=False,
                )

            with gr.Group():
                gr.Markdown("**Optional Parameters**", elem_classes=["optional-params-header"])

            with gr.Row():
                temperature_shift_comp = gr.Textbox(
                    label="Temp. phase shift (rad)",
                    value=(
                        ""
                        if cast("str", _DEFAULTS["temperature_perturbation_phase_shift"]) == ""
                        else str(cast("float", _DEFAULTS["temperature_perturbation_phase_shift"]))
                    ),
                    placeholder="leave empty to use default",
                    lines=1,
                    info="Phase lag between temperature and radial displacement.",
                    interactive=False,
                )

            with gr.Row():
                h_to_r_comp = gr.Textbox(
                    label="H-to-R amplitude ratio",
                    value=(
                        ""
                        if cast("str", _DEFAULTS["horizontal_to_radial_amplitude_ratio"]) == ""
                        else str(cast("float", _DEFAULTS["horizontal_to_radial_amplitude_ratio"]))
                    ),
                    placeholder="leave empty to use default",
                    lines=1,
                    info="Ratio of horizontal to radial amplitude.",
                    interactive=False,
                )

            with gr.Row():
                temperature_amp_comp = gr.Number(
                    label="Temp. amplitude factor",
                    value=float(cast("float", _DEFAULTS["temperature_amplitude_factor"])),
                    info="Ratio dT/T_eff (temperature amplitude).",
                    interactive=False,
                )

            with gr.Row():
                tidally_locked_comp = gr.Checkbox(
                    label="Tidally locked",
                    value=bool(_DEFAULTS["tidally_locked"]),
                    info="Hold mode axis position relative to companion.",
                    interactive=False,
                )

            enabled_comp.change(
                fn=lambda enabled: [gr.update(interactive=bool(enabled)) for _ in range(11)],
                inputs=[enabled_comp],
                outputs=[
                    l_comp,
                    m_comp,
                    amplitude_comp,
                    frequency_comp,
                    start_phase_comp,
                    mode_axis_theta_comp,
                    mode_axis_phi_comp,
                    temperature_shift_comp,
                    h_to_r_comp,
                    temperature_amp_comp,
                    tidally_locked_comp,
                ],
                show_progress="hidden",
                show_progress_on=[],
            )

            components[f"mode_{puls_mode_idx}_enabled"] = enabled_comp
            value_comps[f"mode_{puls_mode_idx}_l"] = l_comp
            value_comps[f"mode_{puls_mode_idx}_m"] = m_comp
            value_comps[f"mode_{puls_mode_idx}_amplitude"] = amplitude_comp
            value_comps[f"mode_{puls_mode_idx}_frequency"] = frequency_comp
            value_comps[f"mode_{puls_mode_idx}_start_phase"] = start_phase_comp
            value_comps[f"mode_{puls_mode_idx}_mode_axis_theta"] = mode_axis_theta_comp
            value_comps[f"mode_{puls_mode_idx}_mode_axis_phi"] = mode_axis_phi_comp
            value_comps[f"mode_{puls_mode_idx}_temperature_perturbation_phase_shift"] = temperature_shift_comp
            value_comps[f"mode_{puls_mode_idx}_horizontal_to_radial_amplitude_ratio"] = h_to_r_comp
            value_comps[f"mode_{puls_mode_idx}_temperature_amplitude_factor"] = temperature_amp_comp
            value_comps[f"mode_{puls_mode_idx}_tidally_locked"] = tidally_locked_comp

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
    if not pulsation_params:
        return []

    modes: list[dict[str, object]] = []
    for puls_mode_parse_idx in range(MAX_PULSE_MODES):
        if not bool(pulsation_params.get(f"mode_{puls_mode_parse_idx}_enabled", False)):
            continue

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
