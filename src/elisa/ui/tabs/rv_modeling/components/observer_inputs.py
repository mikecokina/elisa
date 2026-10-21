"""Gradio component builder for RV observer and sampling parameters.

Provides :func:`build` which renders RV sampling controls inside the
currently active Gradio layout context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from elisa.ui.shared.const import RV_METHODS

if TYPE_CHECKING:
    from elisa.types import Number


# Canonical key order - must stay in sync with the order components are
# yielded by :func:`build` so that ``tab.py`` can reconstruct a named dict
# from a flat value list.
FIELD_ORDER: tuple[str, ...] = (
    "from_phase",
    "to_phase",
    "phase_step",
    "method",
)

_METHOD_DEFAULT: str = RV_METHODS[0]


def build(*, defaults: dict[str, Number | str] | None = None) -> dict[str, gr.Component]:
    """Render RV observer and sampling parameter inputs and return a component mapping.

    Renders numeric inputs for the phase range and step, and a dropdown
    for the RV computation method.

    :param defaults: Optional mapping of field name → default value.
        Unrecognised keys are silently ignored.
    :type defaults: dict[str, Number | str] | None
    :returns: Ordered dict mapping each field name (see :data:`FIELD_ORDER`)
        to its Gradio component.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    components: dict[str, gr.Component] = {}

    gr.Markdown("### Observation Settings")

    with gr.Group():
        components["from_phase"] = gr.Number(
            label="From phase",
            value=defaults.get("from_phase", -0.6),
            info="Start of phase range for RV computation.",
        )
        components["to_phase"] = gr.Number(
            label="To phase",
            value=defaults.get("to_phase", 0.6),
            info="End of phase range for RV computation.",
        )
        components["phase_step"] = gr.Number(
            label="Phase step",
            value=defaults.get("phase_step", 0.01),
            minimum=0.0001,
            info="Phase sampling step (smaller = finer curve, slower computation).",
        )

    components["method"] = gr.Dropdown(
        label="RV computation method",
        choices=RV_METHODS,
        value=_METHOD_DEFAULT,
        info="Kinematic: treats each star as a rigid point mass. "
             "Radiometric: integrates over stellar surface elements.",
    )

    return components

