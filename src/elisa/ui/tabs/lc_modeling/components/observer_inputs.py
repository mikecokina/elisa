"""Gradio component builder for observer and LC sampling parameters.

Provides :func:`build` which renders passband selection and light-curve
sampling controls inside the currently active Gradio layout context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from elisa.conf.settings import settings

if TYPE_CHECKING:
    from elisa.types import Number


# Canonical key order - must stay in sync with the order components are
# yielded by :func:`build` so that ``tab.py`` can reconstruct a named dict
# from a flat value list.
FIELD_ORDER: tuple[str, ...] = (
    "passband",
    "from_phase",
    "to_phase",
    "phase_step",
    "normalize",
)

# Passbands offered in the UI (bolometric excluded from default selection
# because it is rarely used for visual light curves).
_PASSBAND_CHOICES: list[str] = [p for p in settings.PASSBANDS if p != "bolometric"]
_PASSBAND_DEFAULT: list[str] = ["Gaia.2010.G", "Gaia.2010.RP"]


def build(*, defaults: dict[str, Number | str | bool] | None = None) -> dict[str, gr.Component]:
    """Render observer and sampling parameter inputs and return a component mapping.

    Renders a multi-select ``gr.Dropdown`` for passband selection, numeric
    inputs for the phase range and step, and a checkbox for flux normalization.

    :param defaults: Optional mapping of field name → default value.
        Unrecognised keys are silently ignored.
    :type defaults: dict[str, Number | str | bool] | None
    :returns: Ordered dict mapping each field name (see :data:`FIELD_ORDER`)
        to its Gradio component.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    components: dict[str, gr.Component] = {}

    gr.Markdown("### Observation Settings")

    components["passband"] = gr.Dropdown(
        label="Passbands",
        choices=_PASSBAND_CHOICES,
        value=defaults.get("passband", _PASSBAND_DEFAULT),
        multiselect=True,
        info="Select one or more photometric passbands (searchable).",
    )

    with gr.Group():
        components["from_phase"] = gr.Number(
            label="From phase",
            value=defaults.get("from_phase", -0.6),
            info="Start of phase range.",
        )
        components["to_phase"] = gr.Number(
            label="To phase",
            value=defaults.get("to_phase", 0.6),
            info="End of phase range.",
        )
        components["phase_step"] = gr.Number(
            label="Phase step",
            value=defaults.get("phase_step", 0.01),
            minimum=0.0001,
            info="Phase sampling step (smaller = finer, slower).",
        )

    components["normalize"] = gr.Checkbox(
        label="Normalize flux",
        value=defaults.get("normalize", False),
        info="Normalize light curves to a maximum of 1.  "
             "When enabled, distance is not required.",
    )

    return components
