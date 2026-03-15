"""Top-level Gradio application factory for ELISa UI."""

from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr

from elisa.ui.tabs.lc_fitting import tab as lc_fit_tab
from elisa.ui.tabs.lc_modeling import tab as lc_tab
from elisa.ui.tabs.rv_fitting import tab as rv_fit_tab
from elisa.ui.tabs.rv_modeling import tab as rv_tab
from elisa.ui.tabs.system_visualization import tab as sys_viz_tab

_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"


def _get_logo_base64() -> str | None:
    """Load logo image and convert to base64 data URI.

    :returns: Base64-encoded data URI for the logo, or None if file not found.
    :rtype: str | None
    """
    if not _LOGO_PATH.exists():
        return None
    logo_bytes = _LOGO_PATH.read_bytes()
    logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")
    return f"data:image/png;base64,{logo_b64}"


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
        with gr.Row():
            with gr.Column(scale=0, min_width=110):
                logo_data = _get_logo_base64()
                if logo_data:
                    gr.HTML(
                        f'<img src="{logo_data}" alt="ELISa Logo" '
                        f'style="height: 80px; width: 80px; object-fit: contain;" />',
                    )
            with gr.Column(scale=0, min_width=600):
                gr.Markdown(
                    "# ELISa - Binary Star System Modeler\n"
                    "Interactive tool for synthetic light-curve, radial-velocity, "
                    "and system visualization modeling.",
                )
            gr.Column(scale=1)  # Empty column to push content left

        lc_tab.build()
        rv_tab.build()
        rv_fit_tab.build()
        lc_fit_tab.build()
        sys_viz_tab.build()

    return demo
