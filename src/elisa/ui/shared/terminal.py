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
        )

        # Set up clear handler
        def clear_and_refresh() -> str:
            """Clear the buffer and return the updated output."""
            UILogger.clear_output()
            return UILogger.get_output()

        clear_btn.click(
            fn=clear_and_refresh,
            outputs=output_code,
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
        )

        def clear_and_refresh() -> str:
            UILogger.clear_output()
            return UILogger.get_output()

        clear_btn.click(
            fn=clear_and_refresh,
            outputs=output_code,
        )

    return output_code


def setup_terminal_refresh(
    output_component: gr.Code,
    *,
    trigger_component: gr.Component | None = None,
    every_seconds: int | None = None,
) -> None:
    """Set up automatic refresh of terminal output component.

    Can refresh either on a component change or on a timer.

    :param output_component: The Code component to update.
    :type output_component: gr.Code
    :param trigger_component: Component whose changes trigger refresh.
        If None, uses timed refresh.
    :type trigger_component: gr.Component | None
    :param every_seconds: Seconds between refreshes (for timer-based refresh).
        If None, refresh only on component changes.
    :type every_seconds: int | None
    :returns: ``None``
    :rtype: None
    """
    if trigger_component is not None:
        trigger_component.change(
            fn=UILogger.get_output,
            outputs=output_component,
            trigger_mode="always_last",
        )

    if every_seconds is not None:
        gr.Timer(
            value=every_seconds,
        ).tick(
            fn=UILogger.get_output,
            outputs=output_component,
        )

