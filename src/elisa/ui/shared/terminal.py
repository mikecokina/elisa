"""Terminal-like output viewer component for displaying logs in the Gradio UI."""

from __future__ import annotations

import gradio as gr

from elisa.ui.shared.logger import UILogger


def build_terminal_output() -> tuple[gr.Code, gr.Button, gr.Button]:
    """Build a terminal output display component with refresh and clear buttons.

    Creates a read-only Code component styled to look like a terminal,
    with buttons to refresh and clear the output.

    :returns: Tuple of (Code component, Refresh button, Clear button).
    :rtype: tuple[gr.Code, gr.Button, gr.Button]
    """
    with gr.Group():
        gr.Markdown("#### Terminal Output")
        output_code = gr.Code(
            value=UILogger.get_output(),
            language="shell",
            label="",
            interactive=False,
            show_label=False,
            lines=15,
            max_lines=15,
            scale=1,
        )
        with gr.Row():
            refresh_btn = gr.Button(
                "🔄 Refresh",
                scale=0,
                min_width=120,
            )
            clear_btn = gr.Button(
                "🗑️ Clear",
                scale=0,
                min_width=100,
            )

        # Set up refresh handler
        refresh_btn.click(
            fn=UILogger.get_output,
            outputs=output_code,
            show_progress="hidden",
            show_progress_on=[],
        )

        # Set up clear handler
        def clear_and_refresh() -> str:
            """Clear the buffer and return the updated output."""
            UILogger.clear_output()
            return UILogger.get_output()

        clear_btn.click(
            fn=clear_and_refresh,
            outputs=output_code,
            show_progress="hidden",
            show_progress_on=[],
        )

    return output_code, refresh_btn, clear_btn


def build_auto_refresh_terminal_output() -> gr.Code:
    """Build a terminal output component with refresh and clear buttons.

    Creates a read-only Code component styled to look like a terminal,
    with buttons to refresh and clear the output. Auto-refreshes via timer.

    :returns: The Code component displaying terminal output.
    :rtype: gr.Code
    """
    with gr.Group():
        gr.Markdown("#### Terminal Output")
        output_code = gr.Code(
            value=UILogger.get_output(),
            language="shell",
            label="",
            interactive=False,
            show_label=False,
            lines=15,
            max_lines=15,
            scale=1,
        )
        with gr.Row():
            refresh_btn = gr.Button(
                "🔄 Refresh",
                scale=0,
                min_width=120,
            )
            clear_btn = gr.Button(
                "🗑️ Clear",
                scale=0,
                min_width=100,
            )

        refresh_btn.click(
            fn=UILogger.get_output,
            outputs=output_code,
            show_progress="hidden",
            show_progress_on=[],
        )

        def clear_and_refresh() -> str:
            UILogger.clear_output()
            return UILogger.get_output()

        clear_btn.click(
            fn=clear_and_refresh,
            outputs=output_code,
            show_progress="hidden",
            show_progress_on=[],
        )

    return output_code


def setup_terminal_refresh(
    output_component: gr.Code,
    *,
    every_seconds: int | None = None,
) -> None:
    """Set up automatic refresh of terminal output component.

    Refreshes the output component on a timer.

    :param output_component: The Code component to update.
    :type output_component: gr.Code
    :param every_seconds: Seconds between refreshes (for timer-based refresh).
        If None, no automatic refresh is set up.
    :type every_seconds: int | None
    :returns: ``None``
    :rtype: None
    """
    if every_seconds is not None:
        gr.Timer(
            value=every_seconds,
        ).tick(
            fn=UILogger.get_output,
            outputs=output_component,
            show_progress="hidden",
            show_progress_on=[],
        )
