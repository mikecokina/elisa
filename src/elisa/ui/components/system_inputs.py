"""Gradio component builder for binary-system parameters.

Provides :func:`build` which renders the form fields for a
:class:`~elisa.binary_system.system.BinarySystem` inside the currently
active Gradio layout context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from elisa.types import Number

# Canonical key order - must stay in sync with the order components are
# yielded by :func:`build` so that ``tab.py`` can reconstruct a named dict
# from a flat value list.
FIELD_ORDER: tuple[str, ...] = (
    "inclination",
    "period",
    "eccentricity",
    "argument_of_periastron",
    "gamma",
    "phase_shift",
    "additional_light",
    "primary_minimum_time",
    "distance",
)


def build(*, defaults: dict[str, Number | str] | None = None) -> dict[str, gr.Component]:
    """Render binary-system parameter inputs and return a component mapping.

    Mandatory inputs are shown at the top level; optional inputs are
    grouped in a collapsible ``gr.Accordion``.  The function must be
    called inside an active Gradio layout context.

    :param defaults: Optional mapping of field name → default value.
        Unrecognised keys are silently ignored.
    :type defaults: dict[str, object] | None
    :returns: Ordered dict mapping each field name (see :data:`FIELD_ORDER`)
        to its Gradio component.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    components: dict[str, gr.Component] = {}

    gr.Markdown("### Binary System")

    # ------------------------------------------------------------------ #
    # Mandatory parameters                                                 #
    # ------------------------------------------------------------------ #
    components["inclination"] = gr.Number(
        label="Inclination (deg)",
        value=defaults.get("inclination", 85.0),
        minimum=0.0,
        maximum=180.0,
        info="Orbital inclination in degrees.",
    )
    components["period"] = gr.Number(
        label="Period (days)",
        value=defaults.get("period", 2.5),
        minimum=0.0,
        info="Orbital period in days.",
    )
    components["eccentricity"] = gr.Number(
        label="Eccentricity",
        value=defaults.get("eccentricity", 0.0),
        minimum=0.0,
        maximum=0.99,
        info="Orbital eccentricity in [0, 1).",
    )
    components["argument_of_periastron"] = gr.Number(
        label="Argument of periastron (deg)",
        value=defaults.get("argument_of_periastron", 90.0),
        minimum=0.0,
        maximum=360.0,
        info="Argument of periastron in degrees.",
    )

    # ------------------------------------------------------------------ #
    # Optional parameters                                                  #
    # ------------------------------------------------------------------ #
    with gr.Accordion("Binary System - optional parameters", open=False):
        components["gamma"] = gr.Textbox(
            label="Gamma velocity (km/s)",
            value=str(defaults["gamma"]) if "gamma" in defaults else "",
            placeholder="Leave empty for 0.0 (ELISa default)",
            info="Centre-of-mass (systemic) radial velocity in km/s.",
        )
        components["phase_shift"] = gr.Textbox(
            label="Phase shift",
            value=str(defaults["phase_shift"]) if "phase_shift" in defaults else "",
            placeholder="Leave empty for 0.0 (ELISa default)",
            info="Shift applied to phase so that primary minimum coincides with phase 0.",
        )
        components["additional_light"] = gr.Textbox(
            label="Additional light  [0, 1]",
            value=str(defaults["additional_light"]) if "additional_light" in defaults else "",
            placeholder="Leave empty for 0.0 (ELISa default)",
            info="Fraction of total light not originating from the binary system.",
        )
        components["primary_minimum_time"] = gr.Textbox(
            label="Primary minimum time (JD)",
            value=str(defaults["primary_minimum_time"]) if "primary_minimum_time" in defaults else "",
            placeholder="Leave empty for 0.0 (ELISa default)",
            info="Reference time of primary minimum in Julian Days.",
        )
        components["distance"] = gr.Textbox(
            label="Distance (pc)  > 0",
            value=str(defaults["distance"]) if "distance" in defaults else "",
            placeholder="Leave empty - required only for absolute flux",
            info="Distance to the system in parsecs.  Required when normalize is off.",
        )

    return components
