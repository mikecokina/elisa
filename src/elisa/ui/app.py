"""Top-level Gradio application factory for ELISa UI.

Assembles all registered tabs into a single ``gr.Blocks`` application.
Add new tabs by importing their ``build`` function and calling it inside
:func:`build_app`.

Example usage::

    from elisa.ui.app import build_app

    demo = build_app()
    demo.launch()
"""

from __future__ import annotations

import gradio as gr

from elisa.ui.tabs.lc_modeling import tab as lc_tab
from elisa.ui.tabs.rv_modeling import tab as rv_tab


def build_app() -> gr.Blocks:
    """Assemble and return the ELISa Gradio application.

    Creates a ``gr.Blocks`` instance, applies the shared theme and
    header, then delegates to each tab module's ``build`` function.
    New tabs should be registered here by calling their ``build``
    function after the existing ones.

    :returns: Fully configured ``gr.Blocks`` application ready to launch.
    :rtype: gr.Blocks
    """
    with gr.Blocks(
        title="ELISa - Binary Star System Modeler",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# ELISa - Binary Star System Modeler\n"
            "Interactive tool for synthetic light-curve and radial-velocity modeling.",
        )

        # --- register tabs ---
        lc_tab.build()
        rv_tab.build()
        # Future tabs (e.g. fitting) can be added here:
        # fitting_tab.build()

    return demo
