"""Gradio component builder for stellar parameters.

Provides :func:`build` which renders a self-contained form for a single
stellar component (primary or secondary) inside whatever Gradio layout
context the caller has established.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from elisa.ui.shared.const import ATMOSPHERE_CHOICES

if TYPE_CHECKING:
    from elisa.types import Number

# Canonical key order - must stay in sync with the order components are
# yielded by :func:`build` so that ``tab.py`` can reconstruct a named dict
# from a flat value list.
FIELD_ORDER: tuple[str, ...] = (
    "mass",
    "t_eff",
    "surface_potential",
    "synchronicity",
    "gravity_darkening",
    "albedo",
    "metallicity",
    "discretization_factor",
    "atmosphere",
)


def build(prefix: str, *, defaults: dict[str, Number | str] | None = None) -> dict[str, gr.Component]:
    """Render star parameter inputs and return a mapping of components.

    All inputs are rendered inside the currently active Gradio layout
    context (caller is responsible for wrapping this in a ``gr.Column``
    or similar).  Mandatory inputs (mass, t_eff, surface_potential,
    synchronicity) are placed at the top level; optional inputs are
    grouped inside a collapsible ``gr.Accordion``.

    :param prefix: Human-readable label used in the accordion header,
        e.g. ``"Primary Star"`` or ``"Secondary Star"``.
    :type prefix: str
    :param defaults: Optional mapping of field name → default value used
        to pre-populate the inputs.  Unrecognised keys are ignored.
    :type defaults: dict[str, object] | None
    :returns: Ordered dict mapping each field name (see :data:`FIELD_ORDER`)
        to its Gradio component.
    :rtype: dict[str, gr.Component]
    """
    if defaults is None:
        defaults = {}

    components: dict[str, gr.Component] = {}

    gr.Markdown(f"### {prefix}")

    # ------------------------------------------------------------------ #
    # Mandatory parameters                                                 #
    # ------------------------------------------------------------------ #
    components["mass"] = gr.Number(
        label="Mass (M☉)",
        value=defaults.get("mass", 1.0),
        minimum=0.01,
        info="Stellar mass in solar masses.",
    )
    components["t_eff"] = gr.Number(
        label="T_eff (K)",
        value=defaults.get("t_eff", 6000),
        minimum=1000,
        info="Effective surface temperature in Kelvin.",
    )
    components["surface_potential"] = gr.Number(
        label="Surface potential",
        value=defaults.get("surface_potential", 4.0),
        info="Generalised Roche-lobe surface potential (Wilson 1979).",
    )
    components["synchronicity"] = gr.Number(
        label="Synchronicity",
        value=defaults.get("synchronicity", 1.0),
        minimum=0.0,
        info="Rotation-to-orbital frequency ratio (1 = synchronous).",
    )

    # ------------------------------------------------------------------ #
    # Optional parameters                                                  #
    # ------------------------------------------------------------------ #
    with gr.Accordion(f"{prefix} - optional parameters", open=False):
        components["gravity_darkening"] = gr.Textbox(
            label="Gravity darkening  [0, 1]",
            value=str(defaults["gravity_darkening"]) if "gravity_darkening" in defaults else "",
            placeholder="leave empty to auto-interpolate (Claret 2003)",
            info="Gravity-darkening exponent in [0, 1].  Leave empty to interpolate.",
        )
        components["albedo"] = gr.Textbox(
            label="Albedo  [0, 1]",
            value=str(defaults["albedo"]) if "albedo" in defaults else "",
            placeholder="leave empty to auto-interpolate (Claret 2001)",
            info="Surface albedo in [0, 1].  Leave empty to interpolate.",
        )
        components["metallicity"] = gr.Number(
            label="Metallicity [M/H]",
            value=defaults.get("metallicity", 0.0),
            info="Metallicity log [M/H].  Default is 0.0.",
        )
        components["discretization_factor"] = gr.Textbox(
            label="Discretization factor (deg)",
            value=str(defaults["discretization_factor"]) if "discretization_factor" in defaults else "",
            placeholder="leave empty for ELISa default",
            info="Surface mesh angular element size in degrees.  Must be a positive integer.",
        )
        components["atmosphere"] = gr.Dropdown(
            label="Atmosphere model",
            choices=ATMOSPHERE_CHOICES,
            value=defaults.get("atmosphere", ATMOSPHERE_CHOICES[1]),
            info="Atmosphere model used for intensity integration.",
        )

    return components
