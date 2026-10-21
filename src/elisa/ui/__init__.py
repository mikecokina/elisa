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
    theme_mode: Literal["light", "dark", "system"] | None = None,
    port: int | None = None,
    **kwargs: Any,
) -> None:
    """Build and launch the ELISa Gradio UI.

    :param theme_mode: Color scheme to enforce on page load.
        ``"light"`` forces light theme, ``"dark"`` forces dark theme,
        ``"system"`` defers to browser/OS preference.
        If ``None``, reads from ``ELISA_UI_THEME`` environment variable
        (defaults to ``"light"`` if not set). Can also be passed via
        ``kwargs["theme"]`` as a fallback.
    :type theme_mode: Literal["light", "dark", "system"] | None
    :param port: Optional TCP port to bind the Gradio server to. When set,
        this value is forwarded to ``gradio.Blocks.launch`` as
        ``server_port``. If ``None``, reads from ``ELISA_UI_SERVER_PORT``
        environment variable or uses a caller-provided ``server_port`` in
        ``**kwargs``, or Gradio will pick a default.
    :type port: int | None
    :param kwargs: Additional arguments forwarded to ``gr.Blocks.launch``
        (e.g. ``share``, ``inbrowser``, ``server_name``, ``server_port``,
        ``theme``). Environment variables ``ELISA_UI_SERVER_HOST`` and
        ``ELISA_UI_SERVER_PORT`` can also be used to set ``server_name``
        and ``server_port``.
    :type kwargs: object
    :returns: ``None``
    :rtype: None
    """
    import warnings  # noqa: PLC0415

    import gradio as gr  # noqa: PLC0415

    # Resolve theme_mode: explicit parameter > env variable > kwargs > default
    if theme_mode is None:
        theme_mode = os.environ.get("ELISA_UI_THEME", "").lower()
        if not theme_mode:
            # Check if theme was passed via kwargs; use system if present, else light
            theme_mode = "system" if "theme" in kwargs else "light"
        # Validate environment variable value
        if theme_mode not in ("light", "dark", "system"):
            msg = (
                f"Invalid ELISA_UI_THEME value: {theme_mode!r}. "
                "Must be one of: 'light', 'dark', 'system'. Defaulting to 'light'."
            )
            warnings.warn(msg, stacklevel=2)
            theme_mode = "light"

    kwargs.setdefault("theme", gr.themes.Default())
    kwargs.setdefault("css", APP_CSS)

    # Use official Gradio workaround: force theme via __theme URL parameter
    if theme_mode in ("light", "dark"):
        kwargs.setdefault("js", _THEME_JS[theme_mode])

    # Resolve server port: explicit parameter > env variable > kwargs
    if port is not None:
        kwargs.setdefault("server_port", int(port))
    else:
        env_port = os.environ.get("ELISA_UI_SERVER_PORT", "").strip()
        if env_port:
            try:
                kwargs.setdefault("server_port", int(env_port))
            except ValueError:
                msg = (
                    f"Invalid ELISA_UI_SERVER_PORT value: {env_port!r}. "
                    "Must be a valid integer. Ignoring."
                )
                warnings.warn(msg, stacklevel=2)

    # Resolve server host: env variable > kwargs
    env_host = os.environ.get("ELISA_UI_SERVER_HOST", "").strip()
    if env_host:
        kwargs.setdefault("server_name", env_host)

    build_app().launch(**kwargs)
