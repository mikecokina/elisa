"""Top-level Gradio application factory for ELISa UI."""

from __future__ import annotations

import gradio as gr

from elisa.ui.tabs.lc_modeling import tab as lc_tab
from elisa.ui.tabs.rv_fitting import tab as rv_fit_tab
from elisa.ui.tabs.rv_modeling import tab as rv_tab
from elisa.ui.tabs.system_visualization import tab as sys_viz_tab


def build_app() -> gr.Blocks:
    """Assemble and return the ELISa Gradio application.

    Creates a ``gr.Blocks`` instance and delegates to each tab module.
    To enforce a colour scheme, pass ``theme_mode`` to ``launch()`` -
    it will inject the appropriate CSS to override system preferences.

    :returns: Fully configured ``gr.Blocks`` application ready to launch.
    :rtype: gr.Blocks
    """
    with gr.Blocks(
        title="ELISa - Binary Star System Modeler",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# ELISa - Binary Star System Modeler\n"
            "Interactive tool for synthetic light-curve, radial-velocity, "
            "and system visualization modeling.",
        )

        lc_tab.build()
        rv_tab.build()
        rv_fit_tab.build()
        sys_viz_tab.build()

    return demo
