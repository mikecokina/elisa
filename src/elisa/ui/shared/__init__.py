"""Shared helpers for the ELISa Gradio UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from collections.abc import Sequence


def build_full_width_button_row(
    label: str,
    *,
    variant: str = "secondary",
    size: str = "sm",
    scale: int = 1,
    elem_classes: Sequence[str] | None = None,
    spacer_margin_px: int = 8,
) -> gr.Button:
    """Create a full-width button row with a small vertical spacer below.

    The helper assumes it is called inside an active Gradio layout context
    (for example, within a ``Blocks`` or ``Column``). It renders a
    ``gr.Row`` that contains a single full-width :class:`gr.Button` followed
    by a tiny HTML spacer used as visual separation from the content below.

    Parameters
    ----------
    label:
        Button label text.
    variant:
        Gradio style variant for the button.
    size:
        Gradio size preset to use for the button.
    scale:
        Flex scale factor applied to the button inside its row. A value of
        ``1`` makes the button expand to the full available width of the
        row.
    elem_classes:
        Optional sequence of CSS utility classes applied to the button.
    spacer_margin_px:
        Bottom margin, in pixels, applied to the HTML spacer element.

    Returns
    -------
    gr.Button
        The created button instance, suitable for event wiring.

    """
    with gr.Row():
        button = gr.Button(
            label,
            variant=variant,
            size=size,
            scale=scale,
            elem_classes=list(elem_classes) if elem_classes is not None else None,
        )

    gr.HTML(f"<div style='margin-bottom: {spacer_margin_px}px;'></div>")

    return button
