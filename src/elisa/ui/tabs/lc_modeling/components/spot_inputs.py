"""Gradio component builder for star spot inputs used in LC Modeling.

Each star can optionally have several surface spots. The component exposes an
"Add spot" button and per-spot controls; spots are represented as a flat
sequence of Gradio components and parsed back into a list of spot dicts by
``parse_spots``.

Default units and typical values follow examples in ``spots.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import gradio as gr

from elisa.ui.shared import build_full_width_button_row

if TYPE_CHECKING:
    from collections.abc import Callable


# Public constants
MAX_SPOTS: int = 6

_SPOT_PARAMS: tuple[str, ...] = (
    "longitude",
    "latitude",
    "angular_radius",
    "temperature_factor",
)

FIELD_ORDER: tuple[str, ...] = (
    "enabled",
    "spot_count",
    *(f"spot_{i}_{p}" for i in range(MAX_SPOTS) for p in _SPOT_PARAMS),
)

# Sensible defaults inspired by /spots.py example
_DEFAULTS: dict[str, float] = {
    "longitude": 0.0,
    "latitude": 45.0,
    "angular_radius": 15.0,
    "temperature_factor": 1.0,
}


def _make_add_handler(max_spots: int) -> Callable[[int], list[object]]:
    def handler(spot_count: int) -> list[object]:
        current = int(spot_count)
        new_count = min(current + 1, max_spots)
        return [new_count, *[gr.update(visible=(i < new_count)) for i in range(max_spots)]]

    return handler


def _make_remove_handler(slot_idx: int, max_spots: int) -> Callable[..., list[object]]:
    n = len(_SPOT_PARAMS)
    default_row = [float(_DEFAULTS[p]) for p in _SPOT_PARAMS]

    def handler(spot_count: int, *flat_values: object) -> list[object]:
        current = int(spot_count)
        total_outputs = 1 + max_spots + max_spots * n
        if current == 0:
            return [gr.update()] * total_outputs

        new_count = max(0, current - 1)

        spots = [list(flat_values[k * n : (k + 1) * n]) for k in range(max_spots)]

        for k in range(slot_idx, max_spots - 1):
            spots[k] = spots[k + 1]
        spots[-1] = list(default_row)

        out: list[object] = [new_count]
        out.extend(gr.update(visible=(k < new_count)) for k in range(max_spots))
        for k in range(max_spots):
            out.extend(gr.update(value=spots[k][j]) for j in range(n))
        return out

    return handler


def build(prefix: str) -> dict[str, gr.Component]:
    """Render an optional spots section for one stellar component.

    :param prefix: Human-readable prefix ("Primary" / "Secondary").
    :returns: Mapping of FIELD_ORDER keys to components.
    """
    components: dict[str, gr.Component] = {}
    value_comps: dict[str, gr.Component] = {}

    with gr.Accordion(f"{prefix} Spots", open=False):
        enabled_comp = gr.Checkbox(
            label=f"Enable {prefix.lower()} spots",
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

            for spot_slot_idx in range(MAX_SPOTS):
                with gr.Column(visible=False) as grp:
                    if spot_slot_idx > 0:
                        gr.HTML("<hr style='margin: 10px 0;'>")

                    gr.Markdown(f"**Spot {spot_slot_idx + 1}**", elem_classes=["spot-header"])

                    remove_btn_i = gr.Button(
                        "✕ Remove spot",
                        variant="stop",
                        size="sm",
                        scale=1,
                    )

                    with gr.Row():
                        value_comps[f"spot_{spot_slot_idx}_longitude"] = gr.Number(
                            label="Longitude (deg)",
                            value=float(_DEFAULTS["longitude"]),
                            info="Longitude of spot centre in degrees.",
                        )
                        value_comps[f"spot_{spot_slot_idx}_latitude"] = gr.Number(
                            label="Latitude (deg)",
                            value=float(_DEFAULTS["latitude"]),
                            info="Latitude of spot centre in degrees.",
                        )

                    with gr.Row():
                        value_comps[f"spot_{spot_slot_idx}_angular_radius"] = gr.Number(
                            label="Angular radius (deg)",
                            value=float(_DEFAULTS["angular_radius"]),
                            info="Angular radius of the spot in degrees.",
                        )

                    with gr.Row():
                        value_comps[f"spot_{spot_slot_idx}_temperature_factor"] = gr.Number(
                            label="Temperature factor",
                            value=float(_DEFAULTS["temperature_factor"]),
                            info="T_spot / T_local (e.g. 0.98 or 1.05).",
                        )

                spot_groups.append(grp)
                remove_btns.append(remove_btn_i)

        enabled_comp.change(
            fn=lambda v: gr.update(visible=v),
            inputs=[enabled_comp],
            outputs=[spots_section],
        )

        add_btn.click(
            fn=_make_add_handler(MAX_SPOTS),
            inputs=[spot_count],
            outputs=[spot_count, *spot_groups],
        )

        all_value_comps: list[gr.Component] = [
            value_comps[f"spot_{spot_slot_idx}_{p}"] for spot_slot_idx in range(MAX_SPOTS) for p in _SPOT_PARAMS
        ]

        for spot_remove_idx, rm_btn in enumerate(remove_btns):
            rm_btn.click(
                fn=_make_remove_handler(spot_remove_idx, MAX_SPOTS),
                inputs=[spot_count, *all_value_comps],
                outputs=[spot_count, *spot_groups, *all_value_comps],
            )

    components["enabled"] = enabled_comp
    components["spot_count"] = spot_count
    components.update(value_comps)
    return components


def parse_spots(spot_params: dict[str, object] | None) -> list[dict[str, object]]:
    """Convert flat spot params to a list of spot dicts.

    :param spot_params: Flat dict as produced by :func:`build`.
    :type spot_params: dict[str, object] | None
    :returns: List of spot parameter dicts suitable for ``Star(spots=[...])``.
    :rtype: list[dict[str, object]]
    """
    if not spot_params or not bool(spot_params.get("enabled", False)):
        return []

    count = int(spot_params.get("spot_count", 0) or 0)
    if count <= 0:
        return []

    spots: list[dict[str, object]] = []
    for parse_spot_idx in range(min(count, MAX_SPOTS)):
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
