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
    optional) plus a shared x-axis unit selector and a JSON file upload
    for loading initial parameters from a previous fit.

    :returns: Dict with keys:

        - ``"primary_file"`` - file upload for the primary component RV data
        - ``"secondary_file"`` - file upload for the secondary component RV data
        - ``"x_unit"`` - dropdown for the x-axis unit
        - ``"params_json"`` - file upload for restoring parameters from a previous fit

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

    with gr.Accordion("Load Parameters from Previous Fit", open=False):
        gr.Markdown(
            "Upload a result JSON saved by a previous LSQRT or MCMC run to restore "
            "all parameter values, bounds, and fixed flags into the form below.",
        )
        with gr.Row():
            components["params_json"] = gr.File(
                label="Result JSON",
                file_types=[".json"],
                scale=1,
            )

    return components

