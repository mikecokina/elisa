"""Gradio component builder for visualization observer parameters.

Provides :func:`build` which renders phase, component selection, and frame
of reference controls for mesh and orbit visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from elisa.types import Float


# Canonical key order - must stay in sync with the order components are
# yielded by :func:`build` so that ``tab.py`` can reconstruct a named dict
# from a flat value list.
FIELD_ORDER: tuple[str, ...] = (
    "phase",
    "components_to_plot",
    "frame_of_reference",
)

_COMPONENTS_CHOICES: list[str] = ["both", "primary", "secondary"]
_FRAME_CHOICES: list[str] = ["primary", "barycentric"]


def build(*, defaults: dict[str, Float | str] | None = None) -> dict[str, gr.Component]:
    """Render visualization parameter inputs and return a component mapping.

    Renders a numeric slider for phase selection, a dropdown for component
    selection, and a dropdown for orbital frame of reference.

    :param defaults: Optional mapping of field name - default value.
        Unrecognised keys are silently ignored.
    :type defaults: dict[str, Float | str] | None
    :returns: Ordered dict mapping each field name (see :data:`FIELD_ORDER`)
        to its Gradio component.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    components: dict[str, gr.Component] = {}

    phase_default = defaults.get("phase", 0.0)
    components["phase"] = gr.Slider(
        minimum=-0.5,
        maximum=1.5,
        step=0.01,
        value=phase_default,
        label="Orbital Phase",
        info="Phase at which to display the system geometry",
    )

    comp_default = defaults.get("components_to_plot", "both")
    components["components_to_plot"] = gr.Dropdown(
        choices=_COMPONENTS_CHOICES,
        value=comp_default,
        label="Components to Plot",
        info="Which stellar components to visualize",
    )

    frame_default = defaults.get("frame_of_reference", "primary")
    components["frame_of_reference"] = gr.Dropdown(
        choices=_FRAME_CHOICES,
        value=frame_default,
        label="Orbital Frame",
        info="Reference frame for orbital visualization",
    )

    return components

