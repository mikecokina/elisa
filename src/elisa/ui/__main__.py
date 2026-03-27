"""Allow running the UI with ``python -m elisa.ui``."""

from __future__ import annotations

import os

# Must be set before gradio is imported so that all telemetry hooks
# (HuggingFace analytics + api.gradio.app version-check) are disabled.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from elisa.ui import launch

if __name__ == "__main__":
    launch(inbrowser=True, server_port=7861)
