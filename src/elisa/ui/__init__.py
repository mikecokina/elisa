"""ELISa interactive UI built with Gradio.

Quick-start::

    from elisa.ui import launch
    launch()

Or with custom Gradio launch kwargs::

    from elisa.ui import build_app
    build_app().launch(server_port=7861)
"""

from __future__ import annotations

import os

# Disable Gradio telemetry and version-check pings to external servers.
# This must be set before gradio is imported anywhere in the process.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from elisa.ui.app import build_app

__all__ = ["build_app", "launch"]


def launch(**kwargs: object) -> None:
    """Build and launch the ELISa Gradio UI.

    All keyword arguments are forwarded to :meth:`gr.Blocks.launch`.
    The default theme is ``gr.themes.Ocean()``; pass ``theme=`` to
    override it.

    :param kwargs: Optional arguments passed directly to
        ``gr.Blocks.launch`` (e.g. ``server_port``, ``share``,
        ``inbrowser``, ``theme``).
    :type kwargs: object
    :returns: ``None``
    :rtype: None
    """
    import gradio as gr  # noqa: PLC0415

    kwargs.setdefault("theme", gr.themes.Origin())  # type: ignore[attr-defined]
    build_app().launch(**kwargs)
