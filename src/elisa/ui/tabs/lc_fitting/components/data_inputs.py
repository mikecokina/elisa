"""Gradio component builder for LC fitting data upload rows.

Up to :data:`MAX_PASSBAND_ROWS` passband rows are pre-rendered; only the
first is visible by default.  The add/remove buttons in the returned
:class:`LCDataComponents` must be wired in ``tab.py`` to show/hide the
subsequent rows via a ``gr.State`` counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gradio as gr

from elisa.conf.settings import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PASSBAND_ROWS: int = 8

X_UNIT_CHOICES: list[str] = ["Julian days (JD)", "Phases (dimensionless)"]
X_UNIT_DEFAULT: str = X_UNIT_CHOICES[0]

Y_UNIT_CHOICES: list[str] = ["Flux (dimensionless)", "Magnitude (mag)"]
Y_UNIT_DEFAULT: str = Y_UNIT_CHOICES[0]

#: Available passbands offered in the UI (bolometric excluded - rarely used for fitting).
PASSBAND_CHOICES: list[str] = [p for p in settings.PASSBANDS if p != "bolometric"]
PASSBAND_DEFAULT: str = "Generic.Bessell.V"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass
class LCDataComponents:
    """All Gradio components for the LC data upload section.

    :cvar x_unit: Shared x-axis unit dropdown.
    :cvar passband_count: State holding the number of active passband rows (1-8).
    :cvar row_groups: List of ``gr.Group`` containers - toggle ``visible`` to
        show/hide individual rows.
    :cvar row_passbands: Per-row passband dropdown.
    :cvar row_files: Per-row LC data file upload.
    :cvar row_y_units: Per-row y-axis unit dropdown (flux or magnitude).
    :cvar row_ref_mags: Per-row reference magnitude number - only used when
        y-unit is magnitude.
    :cvar add_btn: Button to reveal the next passband row.
    :cvar remove_btn: Button to hide the last passband row.
    """

    x_unit: gr.Dropdown
    passband_count: gr.State
    row_groups: list[gr.Group] = field(default_factory=list)
    row_passbands: list[gr.Dropdown] = field(default_factory=list)
    row_files: list[gr.File] = field(default_factory=list)
    row_y_units: list[gr.Dropdown] = field(default_factory=list)
    row_ref_mags: list[gr.Number] = field(default_factory=list)
    add_btn: gr.Button | None = None
    remove_btn: gr.Button | None = None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build() -> LCDataComponents:
    """Render the LC data upload section and return all component references.

    Pre-renders :data:`MAX_PASSBAND_ROWS` passband rows inside the active
    Gradio layout context.  Only the first row is visible initially; the
    rest are shown/hidden via add/remove buttons wired in ``tab.py``.

    Each row contains:

    - **Passband** - photometric passband selection dropdown.
    - **File** - whitespace-delimited two- or three-column data file upload.
    - **Y unit** - flux (dimensionless) or magnitude.
    - **Reference magnitude** - required when y-unit is magnitude.

    :returns: Dataclass holding every Gradio component reference needed for
        event wiring and handler input collection.
    :rtype: LCDataComponents
    """
    gr.Markdown("### Light Curve Data")
    gr.Markdown(
        "Upload two-column (or three-column with errors) whitespace-delimited files. "
        "Column order: **x** (time/phase) | **flux or magnitude** | **err** (optional). "
        "Fill **Reference mag.** only when using magnitude data.",
    )

    row_groups: list[gr.Group] = []
    row_passbands: list[gr.Dropdown] = []
    row_files: list[gr.File] = []
    row_y_units: list[gr.Dropdown] = []
    row_ref_mags: list[gr.Number] = []

    with gr.Row():
        x_unit = gr.Dropdown(
            choices=X_UNIT_CHOICES,
            value=X_UNIT_DEFAULT,
            label="X-axis unit",
            info="Unit of the independent variable column in all uploaded files.",
            scale=3,
        )
        with gr.Column(scale=1, min_width=160):
            add_btn = gr.Button("+ Add passband", variant="secondary", size="sm")
            remove_btn = gr.Button("- Remove passband", variant="secondary", size="sm")

    passband_count: gr.State = gr.State(value=1)

    for i in range(MAX_PASSBAND_ROWS):
        with gr.Group(visible=(i == 0)) as grp, gr.Row():
            pb = gr.Dropdown(
                choices=PASSBAND_CHOICES,
                value=PASSBAND_DEFAULT,
                label=f"Passband {i + 1}",
                scale=2,
            )
            f = gr.File(
                label=f"LC data file {i + 1}",
                file_types=[".dat", ".txt", ".csv"],
                scale=3,
            )
            yu = gr.Dropdown(
                choices=Y_UNIT_CHOICES,
                value=Y_UNIT_DEFAULT,
                label="Y unit",
                scale=1,
            )
            rm = gr.Number(
                value=None,
                label="Reference mag.",
                info="Required when Y unit is Magnitude.",
                scale=1,
                minimum=0.0,
            )
        row_groups.append(grp)
        row_passbands.append(pb)
        row_files.append(f)
        row_y_units.append(yu)
        row_ref_mags.append(rm)

    return LCDataComponents(
        x_unit=x_unit,
        passband_count=passband_count,
        row_groups=row_groups,
        row_passbands=row_passbands,
        row_files=row_files,
        row_y_units=row_y_units,
        row_ref_mags=row_ref_mags,
        add_btn=add_btn,
        remove_btn=remove_btn,
    )
