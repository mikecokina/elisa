"""Top-level Gradio application factory for ELISa UI."""

from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr

from elisa.ui.shared.logger import UILogger
from elisa.ui.shared.terminal import build_auto_refresh_terminal_output, setup_terminal_refresh
from elisa.ui.tabs.lc_fitting import tab as lc_fit_tab
from elisa.ui.tabs.lc_modeling import tab as lc_tab
from elisa.ui.tabs.rv_fitting import tab as rv_fit_tab
from elisa.ui.tabs.rv_modeling import tab as rv_tab
from elisa.ui.tabs.system_visualization import tab as sys_viz_tab

_LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

APP_CSS = """
input:disabled,
input[readonly],
textarea[readonly],
textarea:disabled {
    opacity: 0.25 !important;
    background-color: var(--input-background-fill) !important;
    cursor: not-allowed !important;
    color: var(--body-text-color) !important;
    filter: grayscale(60%) !important;
}

.param-section-disabled {
    opacity: 0.6 !important;
    pointer-events: none !important;
    cursor: not-allowed !important;
    filter: grayscale(60%) !important;
}

.section-header {
    margin-top: 0.1rem !important;
    margin-bottom: 0.1rem !important;
    margin-left: 0.2rem !important;
    font-weight: 600 !important;
}

.optional-params-header {
    margin-top: 0.1rem !important;
    margin-bottom: 0.1rem !important;
    margin-left: 0.2rem !important;
    font-weight: 600 !important;
}

/* Client-side disable for visualization observer controls.
   Toggled by JS in the visualization_mode change handler to avoid
   the expensive server round-trip through update_loading_stati_state. */
.viz-control-disabled {
    opacity: 0.6 !important;
    pointer-events: none !important;
    cursor: not-allowed !important;
    filter: grayscale(90%) !important;
}

/* Responsive plot image sizing for modeling tabs.
   Keeps the old gr.Plot-like behavior without hard-coded pixel height. */
.responsive-model-plot,
.responsive-model-plot > div,
.responsive-model-plot .image-container {
    width: 100% !important;
}

.responsive-model-plot img {
    width: 100% !important;
    height: auto !important;
    max-height: none !important;
    object-fit: contain !important;
}
"""


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
    # Initialize logging to capture INFO+ output to buffer
    import logging  # noqa: PLC0415

    UILogger.setup_logging(include_timestamp=True, level=logging.INFO)

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

        with gr.Accordion("📋 Terminal Output", open=True):
            terminal_output = build_auto_refresh_terminal_output()
            setup_terminal_refresh(terminal_output, every_seconds=3)

    return demo
