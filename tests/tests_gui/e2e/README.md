# Real fitting E2E tests

These tests use Chromium to interact with the Gradio UI and run real SciPy/emcee
fits. The instructions below assume Linux/Ubuntu and Bash.

## 1. Prepare Python 3.14

Open a terminal in the repository root, where `pyproject.toml`, `src/` and
`tests/` are located. Activate your Python 3.14 environment, or create one:

```bash
python3.14 -m venv .venv-e2e
source .venv-e2e/bin/activate
python --version
```

The version must be `Python 3.14.x`. If `python3.14` is unavailable, install it
with `venv` support first. Activate the environment again in each new terminal.

## 2. Install dependencies and Chromium

```bash
python -m pip install -e '.[test,ui]'
python -m playwright install chromium
```

Playwright is included in the `test` dependencies. Its Python package and browser
are installed separately. Playwright downloads the matching Chromium version;
you do not need Chrome, ChromeDriver or Selenium.

Run the browser installation as the same user who will run the tests, without
`sudo`. On Linux, browsers are stored in `~/.cache/ms-playwright/` by default.
Repeat the browser installation after updating Playwright. If you use
`PLAYWRIGHT_BROWSERS_PATH`, keep its value the same during installation and tests.

If Chromium reports missing system libraries:

```bash
python -m playwright install-deps chromium
```

This installs system packages and may require administrator privileges.

## 3. Configure scientific assets

The observation files are included in `tests/tests_gui/e2e/data/`:

```text
data/
├── synthetic_bessell_U.dat
└── rv.json
```

Keep these files with the tests. Data formats, system parameters and fitting
settings are listed in [data/README.md](data/README.md).

Atmosphere and limb-darkening tables must also be installed and configured.
If they are already available in the following layout, no additional setup is
needed:

```text
~/.elisa/
├── limbdarkening/
│   └── ld/                 # Extracted limb-darkening CSV files
└── atmosphere/
    └── ck04/               # Extracted atmosphere tables, including ckp00/
```

The LC scenarios require `ck04` atmosphere tables and limb-darkening tables for
`Generic.Bessell.U` and bolometric at metallicity 0. Empty directories or
unextracted archives are insufficient. For installation instructions, see
[atmosphere/README.rst](../../../atmosphere/README.rst) and
[limbdarkening/README.rst](../../../limbdarkening/README.rst).

If the tables are installed elsewhere, configure their common parent directory:

```bash
export ELISA_E2E_ASSETS=/path/to/elisa-assets
```

That directory must contain `limbdarkening/ld/` and `atmosphere/ck04/`.
Do not point the variable directly at `ld` or `ck04`. The tests use this setting
instead of reading your normal ELISA configuration. They do not download tables
or modify your normal configuration.

## 4. Check the browser installation

This check starts Chromium without running a fit:

```bash
python - <<'PYTHON'
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_content("<h1>Chromium works</h1>")
    print(page.locator("h1").inner_text())
    browser.close()
PYTHON
```

## 5. Run the tests

```bash
python -m pytest tests/tests_gui/e2e --run-fitting-e2e -q
```

To run one case:

```bash
python -m pytest tests/tests_gui/e2e --run-fitting-e2e -q -k 'LC-least_squares-Community'
```

Without `--run-fitting-e2e`, these tests are skipped before starting the server
or browser. You do not need to start the app or select a port yourself.
The browser runs without a visible window.

Expect **6 passed, 2 skipped**: the current RV form does not support Standard
fitting. A complete run takes approximately 100 seconds. If Playwright is
missing, pytest may skip the entire module; this does not count as a passing
browser test.

MCMC uses **10 steps, 10 walkers and burn_in=0**, with two free parameters.
These short chains validate the workflow, not convergence. LS uses
`max_nfev=3`; LC surfaces use 10-degree discretization and display models have
40 points. Scientific algorithms and numerical validation remain enabled.

## Results and failures

Results are saved in `artifacts/gui-fitting-e2e/`: timings, fitted JSON, saved
chains, screenshots and application logs. Failures also retain HTML, browser
errors and a Playwright trace. Start with `application.log` and `*-failed.png`.
To inspect a trace:

```bash
python -m playwright show-trace artifacts/gui-fitting-e2e/RV-least_squares-Community-trace.zip
```

The trace file exists only if that case failed. You can delete the artifacts
after reviewing them; the next run creates new results. Save any results you
want to keep before another run, because filenames are reused.

## Manual UI testing

For manual testing, use your normal browser and ELISA configuration. Playwright
and its Chromium installation are not required:

```bash
python -m elisa.ui
```

Open the local URL printed in the terminal. Stop the app with `Ctrl+C`.
Configure atmosphere and limb-darkening paths in your normal ELISA configuration
if they are not already configured. `ELISA_E2E_ASSETS` applies only to E2E tests.
For a short manual MCMC run, use 10 steps, 10 walkers and zero burn-in. Ten
walkers require no more than five free parameters (`nwalkers >= 2 * ndim`).

## Troubleshooting

| Problem | Action |
|---|---|
| Missing `playwright`, `gradio` or `pytest` | Activate the correct environment and run `python -m pip install -e '.[test,ui]'`. |
| Browser executable not found | Run `python -m playwright install chromium` as the same user and in the same environment. |
| Missing browser system libraries | Run `python -m playwright install-deps chromium`. |
| Wrong Python version | Check `python --version` and activate Python 3.14. |
| Missing observation files | Restore `tests/tests_gui/e2e/data/` from version control. |
| Missing scientific tables | Check the asset layout and `ELISA_E2E_ASSETS` setting. |
| All cases skipped | Check the `--run-fitting-e2e` argument and Playwright installation. |

## Fast regression tests

```bash
PYTHONPATH=src MPLBACKEND=Agg python -m pytest tests/tests_gui/test_rv_fitting_parameters.py tests/tests_gui/test_ui_reaudit.py tests/tests_gui/test_lc_ui_chain_loading.py tests/tests_gui/test_ui_system_visualization.py -q
```
