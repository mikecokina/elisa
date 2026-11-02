"""Real Chromium -> Gradio -> SciPy/emcee integration; opt in with --run-fitting-e2e."""

import json
import re
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

expect = pytest.importorskip("playwright.sync_api").expect

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/gui-fitting-e2e"
DATA = Path(__file__).resolve().parent / "data"


class FitRecord(TypedDict, total=False):
    scenario: str
    status: str
    reason: str
    workflow_seconds: float
    backend_seconds: float
    total_seconds: float
    r_squared: float | None


def component(app, ident):
    return app["manifest"]["components"][str(ident)]


def control(page, app, ident):
    return page.locator(component(app, ident)["selector"])


def number(locator, value):
    box = locator.locator("input").first
    expect(box).to_be_enabled()
    box.fill(str(value))
    box.press("Tab")
    expect(box).to_have_value(str(value))


def dropdown(page, locator, value):
    locator.locator("input").click()
    locator.get_by_role("option", name=value, exact=True).click()
    locator.locator("input").press("Escape")
    expect(locator.locator("input")).to_have_value(value)


def open_ancestors(locator):
    for accordion in locator.locator('xpath=ancestor::*[contains(@class,"gr-accordion")]').all():
        content = accordion.locator(':scope > [data-testid="accordion-content"]')
        if content.count() and content.get_attribute("style") == "display: none;":
            accordion.locator(":scope > button").click()


def field(page, app, kind, key):
    return control(page, app, app["manifest"]["callbacks"][kind]["fields"][key])


def set_parameter(page, app, kind, prefix, value=None, mode="fixed", bounds=None, constraint=None):
    value_control = field(page, app, kind, prefix + "_value")
    open_ancestors(value_control)
    if value is not None:
        number(value_control, value)
    dropdown(page, field(page, app, kind, prefix + "_mode"), mode)
    if mode == "free":
        for suffix, bound in zip(("min", "max"), bounds, strict=True):
            number(field(page, app, kind, prefix + "_" + suffix), bound)
    if constraint:
        text = field(page, app, kind, prefix + "_constraint").locator("textarea")
        expect(text).to_be_enabled()
        text.fill(constraint)
        text.press("Tab")
    if mode != "free":
        expect(field(page, app, kind, prefix + "_min").locator("input")).to_be_disabled()


def prepare_data(work, kind):
    if kind == "LC":
        data = np.loadtxt(DATA / "synthetic_bessell_U.dat")
        # Copy 20 evenly spaced existing observations, preserving JD, flux and errors.
        indices = np.linspace(0, len(data) - 1, 20, dtype=int)
        target = work / "lc_subset.dat"
        np.savetxt(target, data[indices], fmt="%.12g")
        return [target]
    data = json.loads((DATA / "rv.json").read_text())
    phases = np.arange(-0.6, 0.62, 0.02)
    indices = np.arange(0, len(phases), 4)
    paths = []
    for star, velocities in data.items():
        target = work / f"rv_{star}.dat"
        np.savetxt(
            target,
            np.column_stack((phases[indices], np.asarray(velocities)[indices] / 1000, np.full(len(indices), 0.5))),
            fmt="%.12g",
        )
        paths.append(target)
    return paths


def events(app):
    path = app["work"] / "backend.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize(
    ("kind", "method", "approach"),
    [
        pytest.param(
            kind,
            method,
            approach,
            marks=[
                pytest.mark.requires_ld(passbands=("Generic.Bessell.U", "bolometric")),
                pytest.mark.requires_atmosphere(atlas="ck04"),
            ]
            if kind == "LC"
            else [],
        )
        for kind in ("LC", "RV")
        for method in ("least_squares", "mcmc")
        for approach in ("Community", "Standard")
    ],
)
def test_real_fitting(application, browser, kind, method, approach):
    app = application
    scenario = f"{kind}-{method}-{approach}"
    record: FitRecord = {"scenario": scenario}
    start = time.perf_counter()
    context = browser.new_context(viewport={"width": 1600, "height": 1100}, accept_downloads=True)
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    console = []
    page.on("pageerror", lambda error: console.append(str(error)))
    log_offset = app["log"].stat().st_size
    try:
        page.goto(app["url"], wait_until="domcontentloaded")
        page.get_by_role("tab", name=f"{kind} Fitting", exact=True).click()
        if kind == "RV" and approach == "Standard":
            assert (
                page.get_by_text("Fitting approach", exact=True).count() == 0
                or not page.get_by_text("Fitting approach", exact=True).is_visible()
            )
            pytest.skip("RV UI exposes only asini + mass_ratio; no Standard approach selector or mass inputs")
        for label in (
            "Primary Component Parameters",
            "Secondary Component Parameters",
            "MCMC Nuisance Parameter",
            "3 · MCMC settings",
        ):
            header = page.get_by_role("button", name=re.compile("^" + re.escape(label)))
            if header.count():
                header.click()
        callback = app["manifest"]["callbacks"][kind]
        inputs = callback["inputs"]
        data_paths = prepare_data(app["work"], kind)
        if kind == "LC":
            dropdown(page, control(page, app, inputs[0]), "Julian days (JD)")
            upload = next(c for c in inputs if component(app, c)["label"] == "LC data file 1")
            control(page, app, upload).locator("input[type=file]").set_input_files(str(data_paths[0]))
            page.get_by_label(approach, exact=True).check()
            # Exercise both directions, then return to the requested approach.
            page.get_by_label("Standard" if approach == "Community" else "Community", exact=True).check()
            page.get_by_label(approach, exact=True).check()
            for name, value in {
                "inclination": 85,
                "eccentricity": 0,
                "argument_of_periastron": 58,
                "period": 2.5,
                "primary_minimum_time": 54953.5388437,
                "additional_light": 0,
                "phase_shift": 0,
            }.items():
                set_parameter(
                    page,
                    app,
                    kind,
                    "system_" + name,
                    value,
                    mode="free" if name == "inclination" else "fixed",
                    bounds=(84, 86) if name == "inclination" else None,
                )
            for star, temperature, potential in [("primary", 10000, 3.6), ("secondary", 7000, 4)]:
                for name, value in {
                    "t_eff": temperature,
                    "surface_potential": potential,
                    "gravity_darkening": 1,
                    "albedo": 0.6,
                    "synchronicity": 1,
                    "metallicity": 0,
                }.items():
                    mode = "free" if star == "primary" and name == "t_eff" else "fixed"
                    if star == "secondary" and name == "t_eff":
                        mode = "constrained"
                    set_parameter(
                        page,
                        app,
                        kind,
                        star + "_" + name,
                        value,
                        mode=mode,
                        bounds=(9800, 10200) if mode == "free" else None,
                        constraint="0.7 * primary@t_eff" if mode == "constrained" else None,
                    )
            if approach == "Community":
                set_parameter(page, app, kind, "system_mass_ratio", 1.2 / 2.15)
                # Kepler's third law for M1=2.15, M2=1.2, P=2.5 days.
                set_parameter(page, app, kind, "system_semi_major_axis", 11.6)
            else:
                set_parameter(page, app, kind, "primary_mass", 2.15)
                set_parameter(page, app, kind, "secondary_mass", 1.2)
        else:
            for ident, path in zip(inputs[:2], data_paths, strict=True):
                control(page, app, ident).locator("input[type=file]").set_input_files(str(path))
            dropdown(page, control(page, app, inputs[2]), "Phases (dimensionless)")
            for name, value in {
                "eccentricity": 0,
                "asini": 16.445535068958847,
                "mass_ratio": 0.5,
                "argument_of_periastron": 0,
                "gamma": 20,
                "period": 4.5,
                "primary_minimum_time": 0,
                "ln_f": -20,
            }.items():
                free = name in ("asini", "gamma")
                set_parameter(
                    page,
                    app,
                    kind,
                    name,
                    value,
                    mode="constrained" if name == "eccentricity" else "free" if free else "fixed",
                    bounds=(15, 18) if name == "asini" else (18, 22) if name == "gamma" else None,
                    constraint="0 * system@asini" if name == "eccentricity" else None,
                )
        if method == "mcmc":
            nuisance = "nuisance_ln_f" if kind == "LC" else "ln_f"
            set_parameter(page, app, kind, nuisance, -20)
            for label, value in [("Number of walkers", 10), ("Sampling steps", 10), ("Burn-in / Warmup steps", 0)]:
                candidates = [int(i) for i, c in app["manifest"]["components"].items() if c["label"] == label]
                locator = next(
                    control(page, app, i)
                    for i in candidates
                    if page.locator(component(app, i)["selector"]).count()
                    and (i > inputs[0] if kind == "LC" else i < app["manifest"]["callbacks"]["LC"]["inputs"][0])
                )
                open_ancestors(locator)
                number(locator, value)
            chain_ids = [
                int(i) for i, c in app["manifest"]["components"].items() if c["label"] == "Chain file ID / Path"
            ]
            chain_id = next(i for i in chain_ids if control(page, app, i).is_visible())
            control(page, app, chain_id).locator("textarea").fill(str(app["work"] / f"{scenario}-chain.json"))
        before = len(events(app))
        fit_start = time.perf_counter()
        page.get_by_role(
            "button", name="⚡ Run Least Squares" if method == "least_squares" else "🔬 Run MCMC", exact=True
        ).click()
        download = page.get_by_role("button", name=re.compile("Download.*JSON")).or_(
            page.get_by_role("link", name=re.compile("Download.*JSON"))
        )
        error = page.get_by_text(
            re.compile("Your initial parameters contain|It is not allowed for|Invalid syntax or value in constraint")
        )
        expect(download.or_(error)).to_be_visible(timeout=180000)
        assert not error.count(), error.all_text_contents()
        record["workflow_seconds"] = time.perf_counter() - fit_start
        new_events = events(app)[before:]
        fit = next(event for event in new_events if event.get("method") == method and "seconds" in event)
        assert fit["kind"] == kind and "error" not in fit, fit
        record["backend_seconds"] = fit["seconds"]
        record["workflow_seconds"] = time.perf_counter() - fit_start
        assert len(fit["free"]) == 2
        assert all(shape == [20 if kind == "LC" else 16] for shape in fit["data_shapes"].values())
        expected_approach_key = "primary@mass" if approach == "Standard" else "system@mass_ratio"
        assert expected_approach_key in fit["input_keys"]
        result = fit["result"]
        assert result
        for key, value in fit["fixed"].items():
            section, name = key.split("@")
            if section == "nuisance" and method == "least_squares":
                continue  # LS does not use the MCMC noise parameter.
            meta = result[section][name]
            assert meta["fixed"] is True
            assert meta["value"] == pytest.approx(value, rel=1e-12, abs=1e-12), key
        r_squared = result.get("r_squared", {}).get("value")
        record["r_squared"] = r_squared
        if method == "least_squares":
            assert r_squared is not None
            assert r_squared > 0.99
        if method == "mcmc":
            assert fit["chain_shape"] == [100, 2]
            assert fit["chain_finite"]
            assert fit["options"]["burn_in"] == 0 and fit["options"]["nsteps"] == 10
        for section in ("system", "primary", "secondary", "nuisance"):
            for name, meta in result.get(section, {}).items():
                if isinstance(meta, dict) and "value" in meta:
                    assert np.isfinite(meta["value"]), (section, name, meta)
        if kind == "LC":
            assert result["system"]["period"]["value"] == 2.5
            assert result["primary"]["surface_potential"]["value"] == 3.6
            assert abs(result["system"]["inclination"]["value"] - 85) < 1.01
            assert result["secondary"]["t_eff"]["value"] == pytest.approx(
                0.7 * result["primary"]["t_eff"]["value"], rel=1e-10
            )
            if approach == "Standard":
                assert result["primary"]["mass"]["value"] == 2.15
            else:
                assert result["system"]["mass_ratio"]["value"] == pytest.approx(1.2 / 2.15)
        else:
            assert result["system"]["mass_ratio"]["value"] == 0.5
            assert result["system"]["period"]["value"] == 4.5
            assert result["system"]["eccentricity"]["value"] == pytest.approx(0, abs=1e-12)
        plots = [event for event in new_events if "plot" in event]
        assert plots and all(event["finite"] for event in plots), plots
        # Real output is rendered, not merely returned by a callback.
        expect(page.get_by_text("system@period", exact=True)).to_be_visible()
        constrained_name = "secondary@t_eff" if kind == "LC" else "system@eccentricity"
        row = page.get_by_role("row").filter(has=page.get_by_text(constrained_name, exact=True))
        expect(row).to_contain_text("constrained:")
        for key in fit["free"] + list(fit["constraints"]):
            section, name = key.split("@")
            value_row = page.get_by_role("row").filter(has=page.get_by_text(key, exact=True))
            expect(value_row).to_contain_text(f"{result[section][name]['value']:.6g}")
        labels = (
            ["Model fit"]
            if method == "least_squares"
            else ["Model fit (MCMC median)", "Corner plot", "Parameter traces"]
        )
        for label in labels:
            images = [
                control(page, app, int(i))
                for i, c in app["manifest"]["components"].items()
                if c["type"] == "Image" and c["label"] == label
            ]
            image_control = next(image for image in images if image.is_visible())
            image = image_control.locator("img").last
            expect(image).to_be_visible()
            expect(image).to_have_js_property("complete", True)
            assert image.evaluate("(img) => img.naturalWidth > 100 && img.naturalHeight > 100")
        if method == "mcmc":
            chain_path = app["work"] / f"{scenario}-chain.json"
            chain = np.asarray(json.loads(chain_path.read_text())["flat_chain"])
            assert chain.shape == (100, 2) and np.isfinite(chain).all()

        with page.expect_download() as downloaded:
            download.click()
        target = ARTIFACTS / f"{scenario}-result.json"
        downloaded.value.save_as(target)
        assert json.loads(target.read_text()) == result
        log_text = app["log"].read_bytes()[log_offset:].decode(errors="replace")
        assert not re.search(r"Traceback \(most recent call last\)|ERROR|Exception in", log_text), log_text[-5000:]
        assert not console, console
        page.screenshot(path=str(ARTIFACTS / f"{scenario}-passed.png"), full_page=True)
        record["status"] = "passed"
    except (Exception, pytest.skip.Exception) as exc:
        record.update(status="skipped" if isinstance(exc, pytest.skip.Exception) else "failed", reason=str(exc))
        page.screenshot(path=str(ARTIFACTS / f"{scenario}-{record['status']}.png"), full_page=True)
        (ARTIFACTS / f"{scenario}-application.log").write_bytes(app["log"].read_bytes()[log_offset:])
        (ARTIFACTS / f"{scenario}-browser.txt").write_text("\n".join(console))
        (ARTIFACTS / f"{scenario}-page.html").write_text(page.content())
        raise
    finally:
        record["total_seconds"] = time.perf_counter() - start
        (ARTIFACTS / f"{scenario}-timing.json").write_text(json.dumps(record, indent=2))
        if record.get("status") == "failed":
            context.tracing.stop(path=str(ARTIFACTS / f"{scenario}-trace.zip"))
        else:
            context.tracing.stop()
        context.close()
