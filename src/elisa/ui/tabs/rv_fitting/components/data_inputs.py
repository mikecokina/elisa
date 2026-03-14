"""Gradio component builder for RV data file upload."""

from __future__ import annotations

import gradio as gr

# x-axis unit choices exposed to the user
X_UNIT_CHOICES: list[str] = ["Julian days (JD)", "Phases (dimensionless)"]
X_UNIT_DEFAULT: str = X_UNIT_CHOICES[0]

# Column layout constants
COL_PRIMARY = "_col_primary"
COL_SECONDARY = "_col_secondary"


def build() -> dict[str, gr.Component]:
    """Render the RV data upload section and return a component mapping.

    Creates two side-by-side upload areas (primary required, secondary
    optional) plus a shared x-axis unit selector.  A small preview table
    for each file is included so the user can verify the loaded data
    before running a fit.

    :returns: Dict with keys:

        - ``"primary_file"`` - file upload for the primary component RV data
        - ``"secondary_file"`` - file upload for the secondary component RV data
        - ``"x_unit"`` - dropdown for the x-axis unit

    :rtype: dict[str, gr.Component]
    """
    components: dict[str, gr.Component] = {}

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

