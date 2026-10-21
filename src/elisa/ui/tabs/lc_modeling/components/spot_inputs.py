"""Gradio component builder for star spot inputs used in LC Modeling.

Each star can optionally have several surface spots. All spot slots are always
rendered and each slot has its own enable checkbox. Disabled slots are shown
but non-interactive (greyed out). Only enabled slots are serialized by
``parse_spots``.

Default units and typical values follow examples in ``spots.py``.
"""

from __future__ import annotations

from typing import cast

import gradio as gr

from elisa.ui.shared.const import MAX_SPOTS

_SPOT_PARAMS: tuple[str, ...] = (
    "longitude",
    "latitude",
    "angular_radius",
    "temperature_factor",
)

FIELD_ORDER: tuple[str, ...] = (
    *(f"spot_{i}_enabled" for i in range(MAX_SPOTS)),
    *(f"spot_{i}_{p}" for i in range(MAX_SPOTS) for p in _SPOT_PARAMS),
)

# Sensible defaults inspired by /spots.py example
_DEFAULTS: dict[str, float] = {
    "longitude": 0.0,
    "latitude": 45.0,
    "angular_radius": 15.0,
    "temperature_factor": 1.0,
}


def build(prefix: str) -> dict[str, gr.Component]:
    """Render an optional spots section for one stellar component.

    :param prefix: Human-readable prefix ("Primary" / "Secondary").
    :returns: Mapping of FIELD_ORDER keys to components.
    """
    components: dict[str, gr.Component] = {}
    value_comps: dict[str, gr.Component] = {}

    with gr.Accordion(f"{prefix} Spots", open=False):
        for spot_slot_idx in range(MAX_SPOTS):
            if spot_slot_idx > 0:
                gr.HTML("<hr style='margin: 10px 0;'>")

            with gr.Row():
                gr.Markdown(f"**Spot {spot_slot_idx + 1}**", elem_classes=["spot-header"])
                enabled_comp = gr.Checkbox(
                    label="Use",
                    value=False,
                    scale=0,
                    min_width=80,
                )

            longitude_comp = gr.Number(
                label="Longitude (deg)",
                value=float(_DEFAULTS["longitude"]),
                info="Longitude of spot centre in degrees.",
                interactive=False,
            )
            latitude_comp = gr.Number(
                label="Latitude (deg)",
                value=float(_DEFAULTS["latitude"]),
                info="Latitude of spot centre in degrees.",
                interactive=False,
            )
            angular_radius_comp = gr.Number(
                label="Angular radius (deg)",
                value=float(_DEFAULTS["angular_radius"]),
                info="Angular radius of the spot in degrees.",
                interactive=False,
            )
            temperature_factor_comp = gr.Number(
                label="Temperature factor",
                value=float(_DEFAULTS["temperature_factor"]),
                info="T_spot / T_local (e.g. 0.98 or 1.05).",
                interactive=False,
            )

            enabled_comp.change(
                fn=lambda enabled: [gr.update(interactive=bool(enabled)) for _ in range(4)],
                inputs=[enabled_comp],
                outputs=[longitude_comp, latitude_comp, angular_radius_comp, temperature_factor_comp],
                show_progress="hidden",
                show_progress_on=[],
            )

            components[f"spot_{spot_slot_idx}_enabled"] = enabled_comp
            value_comps[f"spot_{spot_slot_idx}_longitude"] = longitude_comp
            value_comps[f"spot_{spot_slot_idx}_latitude"] = latitude_comp
            value_comps[f"spot_{spot_slot_idx}_angular_radius"] = angular_radius_comp
            value_comps[f"spot_{spot_slot_idx}_temperature_factor"] = temperature_factor_comp

    components.update(value_comps)
    return components


def parse_spots(spot_params: dict[str, object] | None) -> list[dict[str, object]]:
    """Convert flat spot params to a list of spot dicts.

    :param spot_params: Flat dict as produced by :func:`build`.
    :type spot_params: dict[str, object] | None
    :returns: List of spot parameter dicts suitable for ``Star(spots=[...])``.
    :rtype: list[dict[str, object]]
    """
    if not spot_params:
        return []

    spots: list[dict[str, object]] = []
    for parse_spot_idx in range(MAX_SPOTS):
        if not bool(spot_params.get(f"spot_{parse_spot_idx}_enabled", False)):
            continue

        long: object = spot_params.get(f"spot_{parse_spot_idx}_longitude")
        lat: object = spot_params.get(f"spot_{parse_spot_idx}_latitude")
        radius: object = spot_params.get(f"spot_{parse_spot_idx}_angular_radius")
        tfac: object = spot_params.get(f"spot_{parse_spot_idx}_temperature_factor")

        spot: dict[str, object] = {
            "longitude": (
                float(cast("float | int", long))
                if long is not None and str(long).strip() != ""
                else float(_DEFAULTS["longitude"])
            ),
            "latitude": (
                float(cast("float | int", lat))
                if lat is not None and str(lat).strip() != ""
                else float(_DEFAULTS["latitude"])
            ),
            "angular_radius": (
                float(cast("float | int", radius))
                if radius is not None and str(radius).strip() != ""
                else float(_DEFAULTS["angular_radius"])
            ),
            "temperature_factor": (
                float(cast("float | int", tfac))
                if tfac is not None and str(tfac).strip() != ""
                else float(_DEFAULTS["temperature_factor"])
            ),
        }
        spots.append(spot)

    return spots
