"""ELISa interactive UI built with Gradio."""

from __future__ import annotations

import os
from typing import Any, Literal

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from elisa.ui.app import APP_CSS, build_app

__all__ = ["build_app", "launch"]

# JavaScript to force theme via URL parameter (official Gradio workaround)
# Uses IIFE (Immediately Invoked Function Expression) to execute on page load
_THEME_JS = {
    "light": """
(function() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
})();
""",
    "dark": """
(function() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
})();
""",
}


def launch(
    *,
    theme_mode: Literal["light", "dark", "system"] = "light",
    port: int | None = None,
    **kwargs: Any,
) -> None:
    """Build and launch the ELISa Gradio UI.

    :param theme_mode: Color scheme to enforce on page load.
        ``"light"`` forces light theme, ``"dark"`` forces dark theme,
        ``"system"`` defers to browser/OS preference.
    :type theme_mode: Literal["light", "dark", "system"]
    :param port: Optional TCP port to bind the Gradio server to. When set,
        this value is forwarded to ``gradio.Blocks.launch`` as
        ``server_port``. If ``None``, the caller may provide ``server_port``
        via ``**kwargs`` or Gradio will pick a default.
    :type port: int | None
    :param kwargs: Additional arguments forwarded to ``gr.Blocks.launch``
        (e.g. ``share``, ``inbrowser``).
    :type kwargs: object
    :returns: ``None``
    :rtype: None
    """
    import gradio as gr  # noqa: PLC0415

    kwargs.setdefault("theme", gr.themes.Default())
    kwargs.setdefault("css", APP_CSS)

    # Use official Gradio workaround: force theme via __theme URL parameter
    if theme_mode in ("light", "dark"):
        kwargs.setdefault("js", _THEME_JS[theme_mode])

    # Allow explicit port override via the `port` parameter. If the caller
    # already provided `server_port` in kwargs, do not override it.
    if port is not None:
        kwargs.setdefault("server_port", int(port))

    build_app().launch(**kwargs)
