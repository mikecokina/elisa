"""Gradio component builder for RV data file upload."""

from __future__ import annotations

from typing import TypedDict

import gradio as gr

# x-axis unit choices exposed to the user
X_UNIT_CHOICES: list[str] = ["Julian days (JD)", "Phases (dimensionless)"]
X_UNIT_DEFAULT: str = X_UNIT_CHOICES[0]

# Column layout constants
COL_PRIMARY = "_col_primary"
COL_SECONDARY = "_col_secondary"


class DataInputComponents(TypedDict):
    """Typed mapping returned by :func:`build`.

    :cvar primary_file: File upload for the primary component RV data.
    :cvar secondary_file: File upload for the secondary component RV data.
    :cvar x_unit: Dropdown for the x-axis unit.
    """

    primary_file: gr.File
    secondary_file: gr.File
    x_unit: gr.Dropdown


def build() -> DataInputComponents:
    """Render the RV data upload section and return a component mapping.

    Creates two side-by-side upload areas (primary required, secondary
    optional) plus a shared x-axis unit selector.

    :returns: Typed dict with keys:

        - ``"primary_file"`` - file upload for the primary component RV data
        - ``"secondary_file"`` - file upload for the secondary component RV data
        - ``"x_unit"`` - dropdown for the x-axis unit

    :rtype: DataInputComponents
    """
    components: DataInputComponents = {}  # type: ignore[typeddict-item]

    gr.Markdown("### Observational Data")
    gr.Markdown(
        "Upload two-column (or three-column with errors) whitespace-delimited files. "
        "Column order: **x** (time/phase) | **RV** (km/s) | **err** (optional).",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Primary component**")
            components["primary_file"] = gr.File(
                label="Primary RV data",
                file_types=[".dat", ".txt", ".csv"],
            )

        with gr.Column(scale=1):
            gr.Markdown("**Secondary component** *(optional)*")
            components["secondary_file"] = gr.File(
                label="Secondary RV data (optional)",
                file_types=[".dat", ".txt", ".csv"],
            )

    with gr.Row():
        components["x_unit"] = gr.Dropdown(
            choices=X_UNIT_CHOICES,
            value=X_UNIT_DEFAULT,
            label="X-axis unit",
            info="Unit of the independent variable column in the uploaded files.",
            scale=1,
        )

    return components
