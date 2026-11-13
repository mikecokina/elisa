"""Temperature increments, plot-local snapshots and zero-amplitude synthesis."""

from __future__ import annotations

import pickle
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from elisa import BinarySystem, Observer, SingleSystem, Star
from elisa import units as u
from elisa.base.container import StarContainer
from elisa.base.error import ElisaGraphicsError
from elisa.base.graphics import plot
from elisa.graphic import graphics
from elisa.pulse import container_ops
from elisa.pulse.surface import kinematics

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from elisa.binary_system.container import OrbitalPositionContainer
    from elisa.pulse.mode import PulsationMode
    from elisa.single_system.container import SinglePositionContainer


SystemKind: TypeAlias = Literal["single", "binary"]
PositionContainer: TypeAlias = "SinglePositionContainer | OrbitalPositionContainer"
ModeParameters: TypeAlias = dict[str, Any]


MODE: ModeParameters = {
    "l": 2,
    "m": 1,
    "amplitude": 500 * u.m / u.s,
    "frequency": 15 / u.d,
    "start_phase": 0.2 * u.rad,
    "mode_axis_theta": 25 * u.deg,
    "mode_axis_phi": 90 * u.deg,
    "horizontal_to_radial_amplitude_ratio": 0.1,
    "temperature_amplitude_factor": 0.005,
    "temperature_perturbation_phase_shift": np.pi / 3 * u.rad,
}
SPOT = {"longitude": 90, "latitude": 80, "angular_radius": 20, "temperature_factor": 0.9}
PHASES = np.array([0.17, 0.2, 0.23])


def make_system(
    kind: SystemKind,
    modes: list[ModeParameters],
    spotted: bool = False,
    *,
    secondary_modes: bool = False,
) -> SingleSystem | BinarySystem:
    features = {"pulsations": deepcopy(modes), "spots": [deepcopy(SPOT)] if spotted else []}
    if kind == "single":
        return SingleSystem(
            star=Star(
                mass=1,
                t_eff=5772,
                polar_log_g=4.43775,
                gravity_darkening=0.32,
                metallicity=0,
                discretization_factor=10,
                atmosphere="ck04",
                **features,
            ),
            rotation_period=25.38,
            inclination=85,
            gamma=0,
            distance=100 * u.pc,
        )
    return BinarySystem(
        primary=Star(
            mass=2.15,
            t_eff=10000,
            surface_potential=5,
            synchronicity=1,
            metallicity=0,
            discretization_factor=5,
            atmosphere="ck04",
            **features,
        ),
        secondary=Star(
            mass=1.2,
            t_eff=7000,
            surface_potential=7,
            synchronicity=1,
            metallicity=0,
            discretization_factor=8,
            atmosphere="ck04",
            **(deepcopy(features) if secondary_modes else {}),
        ),
        argument_of_periastron=90,
        eccentricity=0,
        period=1.7,
        inclination=85,
        primary_minimum_time=2450000,
        distance=100 * u.pc,
    )


def component(container: PositionContainer, kind: SystemKind) -> StarContainer:
    return getattr(container, "star" if kind == "single" else "primary")


def fractional_temperature(star: StarContainer) -> NDArray:
    vertex = sum(
            m.temperature_amplitude_factor
            * np.real(
                np.exp(-1j * m.temperature_perturbation_phase_shift) * m.point_harmonics,
            )
            for m in star.pulsations.values()
        )
    # noinspection PyUnresolvedReferences
    return vertex[star.faces].mean(axis=1)


def thermal_modes(scenario: str) -> list[ModeParameters]:
    modes = [deepcopy(MODE)]
    if scenario == "multiple":
        modes.append({**MODE, "l": 3, "m": -2, "frequency": 11 / u.d, "temperature_amplitude_factor": 0.003})
    elif scenario == "zero_thermal":
        modes[0]["temperature_amplitude_factor"] = 0
    elif scenario in {"zero_velocity", "zero_automatic"}:
        modes[0]["amplitude"] = 0 * u.m / u.s
        if scenario == "zero_automatic":
            modes[0].pop("temperature_amplitude_factor")
    return modes


@pytest.mark.requires_ld
@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize("spotted", [False, True])
def test_temperature_snapshots_across_builds_and_copies(kind: SystemKind, spotted: bool) -> None:
    system = make_system(kind, [MODE], spotted)
    container = system.build_container(phase=0.2)
    original = pickle.dumps(container)
    copied = container.copy()
    for phase in [0.31, 0.48, 0.57]:
        fresh = system.build_container(phase=phase)
        copied.set_on_position_params(fresh.position)
        copied.set_time()
        copied.build(build_pulsations=False)
        equilibrium = cast("NDArray", component(copied, kind).temperatures).copy()
        copied.build_pulsations()
        star = component(copied, kind)
        assert_allclose(
            cast("NDArray", star.temperatures),
            cast("NDArray", component(fresh, kind).temperatures),
            rtol=1e-14,
        )
        before = pickle.dumps(star)
        for _ in range(2):
            assert_array_equal(
                plot.t_cmap(star, "linear", "default", True, equilibrium_temperatures=equilibrium),
                cast("NDArray", star.temperatures) - equilibrium,
            )
            assert_array_equal(plot.t_cmap(star, "linear", "default", False), cast("NDArray", star.temperatures))
        assert pickle.dumps(star) == before
    assert pickle.dumps(container) == original


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize("spotted", [False, True])
@pytest.mark.parametrize("scenario", ["one", "multiple", "zero_thermal", "zero_velocity", "zero_automatic"])
def test_public_temperature_plot_is_repeatable(
    kind: SystemKind,
    spotted: bool,
    scenario: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modes = thermal_modes(scenario)
    system = make_system(kind, modes, spotted)
    original_map = plot.t_cmap
    original_update = container_ops.temp_perturbation
    references = []
    captures = []

    def capture(star: StarContainer, *args: Any, **perturbation_kwargs: Any) -> NDArray | None:
        # Independent observation of the real synthesis boundary, not the
        # plotting implementation's snapshot or a reconstructed harmonic factor.
        if perturbation_kwargs.get("update_container"):
            references.append(cast("NDArray", star.temperatures).copy())
        result = original_update(star, *args, **perturbation_kwargs)
        if perturbation_kwargs.get("update_container"):
            factor = fractional_temperature(star)
            expected = cast("NDArray", star.temperatures) - references[-1]
            assert_allclose(expected, references[-1] * factor, atol=2e-12, rtol=1e-12)
            assert_allclose(
                cast("NDArray", star.temperatures) * factor / (1 + factor),
                expected,
                atol=2e-12,
                rtol=1e-12,
            )
        return result

    def record(star: StarContainer, *args: Any, **colormap_kwargs: Any) -> NDArray:
        colormap_state_before = pickle.dumps(star)
        values = original_map(star, *args, **colormap_kwargs)
        expected = (
            cast("NDArray", star.temperatures) - references[-1] if args[2] else cast("NDArray", star.temperatures)
        )
        assert_array_equal(values, expected)
        assert pickle.dumps(star) == colormap_state_before
        captures.append(values.copy())
        return values

    monkeypatch.setattr(container_ops, "temp_perturbation", capture)
    monkeypatch.setattr(plot, "t_cmap", record)
    before = pickle.dumps(system)
    for phase in [0.2, 0.31]:
        for subtract in [True, False]:
            for _ in range(2):
                kwargs: dict[str, Any] = {"components_to_plot": "primary"} if kind == "binary" else {}
                figure = system.plot.surface(
                    phase=phase,
                    colormap="temperature",
                    subtract_equilibrium=subtract,
                    return_figure_instance=True,
                    **kwargs,
                )
                plt.close(figure)
            assert_array_equal(captures[-1], captures[-2])
    assert pickle.dumps(system) == before


@pytest.mark.requires_ld
@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize("explicit", [False, True])
def test_zero_velocity_thermal_contract(kind: SystemKind, explicit: bool) -> None:
    mode = {**MODE, "amplitude": 0 * u.m / u.s}
    if not explicit:
        mode.pop("temperature_amplitude_factor")
    system = make_system(kind, [mode])
    container = system.build_container(phase=0.2, build_pulsations=False)
    star = component(container, kind)
    equilibrium = cast("NDArray", star.temperatures).copy()
    with np.errstate(divide="raise", invalid="raise"):
        container.build_pulsations()
    assert np.all(np.isfinite(cast("NDArray", star.temperatures)))
    factor = fractional_temperature(star)
    if explicit:
        assert np.max(np.abs(factor)) > 0.001
    else:
        assert_array_equal(factor, 0)
    assert_allclose(cast("NDArray", star.temperatures) - equilibrium, equilibrium * factor, atol=2e-12, rtol=1e-12)
    assert_array_equal(star.pulsations[0].complex_displacement, 0)


def observe(system: SingleSystem | BinarySystem) -> tuple[dict, dict]:
    observer = Observer(system=system, passband=["Generic.Bessell.V"])
    _, flux = observer.lc(phases=PHASES, normalize=False, flux_unit=u.W / u.m**2)
    _, rv = observer.rv(phases=PHASES, method="radiometric")
    return flux, rv


@pytest.mark.requires_ld(passbands=("bolometric", "Generic.Bessell.V"))
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize("explicit", [False, True])
def test_zero_velocity_lc_and_rv(kind: SystemKind, explicit: bool) -> None:
    mode = {**MODE, "amplitude": 0 * u.m / u.s}
    if not explicit:
        mode.pop("temperature_amplitude_factor")
    flux, rv = observe(make_system(kind, [mode]))
    for curve in [*flux.values(), *rv.values()]:
        assert np.all(np.isfinite(curve))


@pytest.mark.requires_ld(passbands=("bolometric", "Generic.Bessell.V"))
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize("explicit", [False, True])
def test_nonzero_lc_rv_match_original_temperature_law(
    kind: SystemKind,
    explicit: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode_parameters = deepcopy(MODE)
    if not explicit:
        mode_parameters.pop("temperature_amplitude_factor")
    system = make_system(kind, [mode_parameters])
    scale = cast("BinarySystem", system).semi_major_axis if kind == "binary" else 1
    flux, rv = observe(system)

    def original_temperature_law(mode: PulsationMode) -> NDArray:
        return (
            mode.temperature_amplitude_factor
            * np.real(
                np.exp(-1j * mode.temperature_perturbation_phase_shift) * mode.complex_radial_displacement,
            )
            * scale
            / mode.radial_amplitude
        )

    monkeypatch.setattr(kinematics, "calculate_temperature_pert_factor", original_temperature_law)
    old_flux, old_rv = observe(deepcopy(system))
    for current, old in [(flux, old_flux), (rv, old_rv)]:
        for key, curve in current.items():
            assert np.all(np.isfinite(curve))
            assert np.all(np.isfinite(old[key]))
            assert_allclose(curve, old[key], rtol=1e-12, atol=1e-10 if current is rv else 0)


def test_reset_flatten_changes_only_flatten_flag() -> None:
    star = StarContainer(temperatures=np.array([6000.0]))
    star._flatten = True
    before = vars(star).copy()
    star.reset_flatten()
    assert star._flatten is False
    assert vars(star).keys() == before.keys()
    for name, value in before.items():
        if name != "_flatten":
            assert getattr(star, name) is value


@pytest.mark.requires_ld
@pytest.mark.parametrize("beta", [1.0, 1 - 1e-12, 1 - 1e-15])
def test_inverse_loses_information_at_accepted_thermal_amplitudes(beta: float) -> None:
    mode = {
        **MODE,
        "l": 0,
        "m": 0,
        "amplitude": 0 * u.m / u.s,
        "start_phase": 0 * u.rad,
        "temperature_perturbation_phase_shift": np.pi * u.rad,
        "temperature_amplitude_factor": beta,
    }
    system = make_system("single", [mode])
    container = system.build_container(phase=0, build_pulsations=False)
    star = cast("SinglePositionContainer", container).star
    equilibrium = cast("NDArray", star.temperatures).copy()
    container.build_pulsations()
    expected = cast("NDArray", star.temperatures) - equilibrium
    assert_array_equal(
        plot.t_cmap(star, "linear", "default", True, equilibrium_temperatures=equilibrium),
        expected,
    )
    factor = fractional_temperature(star)
    # This accepted radial mode is an exact counterexample to inversion;
    # nearby coefficients expose rounding amplified by division by 1 + f.
    if beta == 1:
        assert_array_equal(factor, -1)
        assert_array_equal(cast("NDArray", star.temperatures), 0)
        assert_array_equal(expected, -equilibrium)
    else:
        inverse = cast("NDArray", star.temperatures) * factor / (1 + factor)
        assert np.max(np.abs(inverse - expected)) > 0.1


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("kind", ["single", "binary"])
def test_nonpulsating_public_temperature_plot(kind: SystemKind) -> None:
    system = make_system(kind, [])
    kwargs: dict[str, Any] = {"components_to_plot": "primary"} if kind == "binary" else {}
    before = pickle.dumps(system)
    figure = system.plot.surface(colormap="temperature", return_figure_instance=True, **kwargs)
    plt.close(figure)
    with pytest.raises(ElisaGraphicsError, match="does not oscillate"):
        system.plot.surface(colormap="temperature", subtract_equilibrium=True, **kwargs)
    assert pickle.dumps(system) == before


def test_temperature_subtraction_requires_explicit_snapshot() -> None:
    star = StarContainer(temperatures=np.array([6000.0]))
    with pytest.raises(ElisaGraphicsError, match="pre-pulsation face temperatures"):
        plot.t_cmap(star, "linear", "default", True)


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("kind", ["single", "binary"])
def test_animation_temperature_snapshots(kind: SystemKind, monkeypatch: pytest.MonkeyPatch) -> None:
    system = make_system(kind, thermal_modes("multiple"), spotted=True, secondary_modes=True)
    original_update = container_ops.temp_perturbation
    original_map = plot.t_cmap
    expected = []
    rendered = []

    def capture(star: StarContainer, *args: Any, **kwargs: Any) -> NDArray | None:
        equilibrium = cast("NDArray", star.temperatures).copy()
        result = original_update(star, *args, **kwargs)
        expected.append(cast("NDArray", star.temperatures) - equilibrium)
        return result

    def record(star: StarContainer, *args: Any, **kwargs: Any) -> NDArray:
        colormap_state_before = pickle.dumps(star)
        values = original_map(star, *args, **kwargs)
        assert_array_equal(values, expected.pop(0))
        assert pickle.dumps(star) == colormap_state_before
        return values

    def render(**kwargs: Any) -> None:
        rendered.append(kwargs)

    monkeypatch.setattr(container_ops, "temp_perturbation", capture)
    monkeypatch.setattr(plot, "t_cmap", record)
    monkeypatch.setattr(graphics, f"{kind}_surface_anim", render)
    before = pickle.dumps(system)
    animate = (
        cast("SingleSystem", system).animation.rotational_motion
        if kind == "single"
        else cast("BinarySystem", system).animation.orbital_motion
    )
    animate(start_phase=0.2, stop_phase=0.31, phase_step=0.05, colormap="temperature", subtract_equilibrium=True)
    assert len(rendered) == 1
    assert not expected
    assert pickle.dumps(system) == before


@pytest.mark.requires_ld
@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize("spotted", [False, True])
@pytest.mark.parametrize("scenario", ["one", "multiple", "none"])
def test_complete_build_state_equivalence(kind: SystemKind, spotted: bool, scenario: str) -> None:
    modes = [] if scenario == "none" else thermal_modes(scenario)
    system = make_system(kind, modes, spotted, secondary_modes=True)
    original = system.build_container(phase=0.2, build_pulsations=False)
    split = original.copy()
    for phase in [0.2, 0.31, 0.48]:
        position = system.build_container(phase=phase, build_pulsations=False).position
        for container in [original, split]:
            container.set_on_position_params(position)
            container.set_time()
        original.build(build_pulsations=True)
        split.build(build_pulsations=False)
        split.build_pulsations()
        assert_complete_state_equal(original, split)


@pytest.mark.requires_ld
@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize("phase", [0.2, 0.31])
@pytest.mark.parametrize(
    ("colormap", "subtract"),
    [("temperature", True), ("temperature", False), ("gravity_acceleration", False)],
)
def test_diagnostic_build_preserves_complete_surface(
    kind: SystemKind, colormap: str, subtract: bool, phase: float,
) -> None:
    system = make_system(kind, thermal_modes("multiple"), spotted=True, secondary_modes=True)
    container = system.build_container(phase=phase, build_pulsations=False)
    names = ["star"] if kind == "single" else ["primary", "secondary"]
    equilibrium = {name: getattr(container, name).temperatures.copy() for name in names}
    original = container.copy()
    if kind == "single":
        original.build(phase=phase)
        references = plot.build_surface_with_temperature_reference(
            container, colormap, subtract_equilibrium=subtract,
        )
    else:
        original.build(components_distance=container.position.distance, components="both")
        references = plot.build_surface_with_temperature_reference(
            container, colormap, components_distance=container.position.distance,
            component="all", subtract_equilibrium=subtract,
        )
    assert_complete_state_equal(container, original)
    if colormap == "temperature" and subtract:
        for name in names:
            assert_array_equal(references[name], equilibrium[name])
            assert not np.shares_memory(references[name], getattr(container, name).temperatures)
    else:
        assert references == {}


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("kind", ["single", "binary"])
def test_public_render_matches_unsplit_build(kind: SystemKind, monkeypatch: pytest.MonkeyPatch) -> None:
    system = make_system(kind, thermal_modes("multiple"), spotted=True, secondary_modes=True)
    renderer = "single_star_surface" if kind == "single" else "binary_surface"
    monkeypatch.setattr(graphics, renderer, lambda **kwargs: kwargs)
    split = system.plot.surface(phase=0.31, colormap="temperature", subtract_equilibrium=True)

    def unsplit(container: PositionContainer, _colormap: str, **_kwargs: Any) -> dict:
        names = ["star"] if kind == "single" else ["primary", "secondary"]
        references = {}
        update = container_ops.temp_perturbation

        def capture(star: StarContainer, *args: Any, **options: Any) -> NDArray | None:
            for name in names:
                if getattr(container, name) is star:
                    references[name] = cast("NDArray", star.temperatures).copy()
            return update(star, *args, **options)

        with monkeypatch.context() as patch:
            patch.setattr(container_ops, "temp_perturbation", capture)
            if kind == "single":
                container.build(phase=0.31, build_pulsations=True)
            else:
                container.build(
                    components_distance=container.position.distance, components="both", build_pulsations=True,
                )
        return references

    monkeypatch.setattr(plot, "build_surface_with_temperature_reference", unsplit)
    original = system.plot.surface(phase=0.31, colormap="temperature", subtract_equilibrium=True)
    # Compare the complete renderer input, including coordinates, connectivity,
    # both component colormaps, masks and viewing parameters.
    assert pickle.dumps(split) == pickle.dumps(original)


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize("kind", ["single", "binary"])
@pytest.mark.parametrize("colormap", ["temperature", "gravity_acceleration"])
def test_nonpulsating_animation(kind: SystemKind, colormap: str, monkeypatch: pytest.MonkeyPatch) -> None:
    system = make_system(kind, [])
    frames = []
    monkeypatch.setattr(graphics, f"{kind}_surface_anim", lambda **kwargs: frames.append(kwargs))
    animate = (
        cast("SingleSystem", system).animation.rotational_motion
        if kind == "single"
        else cast("BinarySystem", system).animation.orbital_motion
    )
    before = pickle.dumps(system)
    animate(start_phase=0.2, stop_phase=0.31, phase_step=0.05, colormap=colormap)
    assert len(frames) == 1
    with pytest.raises(ElisaGraphicsError, match="does not oscillate"):
        animate(start_phase=0.2, stop_phase=0.31, phase_step=0.05, colormap=colormap, subtract_equilibrium=True)
    assert pickle.dumps(system) == before


def assert_complete_state_equal(left: object, right: object, path: str = "container") -> None:
    """Compare every stored value exactly, without pickle string-interning artifacts."""
    assert type(left) is type(right), path
    if isinstance(left, np.ndarray):
        assert left.dtype == cast("NDArray", right).dtype, path
        assert_array_equal(left, right, err_msg=path, strict=True)
    elif isinstance(left, dict):
        assert left.keys() == cast("dict[Any, Any]", right).keys(), path
        for key in left:
            assert_complete_state_equal(left[key], cast("dict[Any, Any]", right)[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(cast("tuple | list", right)), path
        for index, value in enumerate(left):
            assert_complete_state_equal(value, cast("tuple | list", right)[index], f"{path}[{index}]")
    elif hasattr(left, "__dict__"):
        assert_complete_state_equal(vars(left), vars(right), path)
    else:
        assert_array_equal(left, right, err_msg=path)


@pytest.mark.requires_ld
@pytest.mark.requires_atmosphere
@pytest.mark.parametrize(
    ("kind", "animation", "explicit_options"),
    [("single", False, {}), ("single", True, {}),
     ("binary", False, {"component": "all"}), ("binary", True, {"component": "all"})],
)
@pytest.mark.parametrize("subtract_temperature", [False, True])
def test_public_graphics_uses_recognized_build_arguments(
    kind: SystemKind,
    animation: bool,
    subtract_temperature: bool,
    explicit_options: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe real graphics calls and reject forwarding of ignored build keywords."""
    system = make_system(kind, thermal_modes("multiple"), spotted=True, secondary_modes=True)
    container_type = type(system.build_container(phase=0.31, build_pulsations=False))
    original_build = container_type.build
    original_pulsations = container_type.build_pulsations
    build_calls: list[tuple[float, float, dict[str, Any]]] = []
    pulsation_calls: list[tuple[float, dict[str, Any]]] = []

    def record_build(
        built_container: PositionContainer, *build_args: Any, **build_options: Any,
    ) -> PositionContainer:
        assert not build_args
        build_calls.append((built_container.position.phase, built_container.position.distance, build_options.copy()))
        return original_build(built_container, *build_args, **build_options)

    def record_pulsations(
        pulsating_container: PositionContainer, *pulse_args: Any, **pulse_options: Any,
    ) -> Any:
        assert not pulse_args
        pulsation_calls.append((pulsating_container.position.distance, pulse_options.copy()))
        return original_pulsations(pulsating_container, *pulse_args, **pulse_options)

    monkeypatch.setattr(container_type, "build", record_build)
    monkeypatch.setattr(container_type, "build_pulsations", record_pulsations)
    if animation:
        monkeypatch.setattr(graphics, f"{kind}_surface_anim", lambda **_frame_options: None)
        animate_system = (
            cast("SingleSystem", system).animation.rotational_motion if kind == "single"
            else cast("BinarySystem", system).animation.orbital_motion
        )
        animate_system(start_phase=0.2, stop_phase=0.31, phase_step=0.05,
                       colormap="temperature", subtract_equilibrium=subtract_temperature)
    else:
        renderer_name = "single_star_surface" if kind == "single" else "binary_surface"
        monkeypatch.setattr(graphics, renderer_name, lambda **_surface_options: None)
        system.plot.surface(phase=0.31, colormap="temperature", subtract_equilibrium=subtract_temperature)

    assert len(build_calls) == (2 if animation else 1)
    assert len(pulsation_calls) == len(build_calls)
    for _built_phase, built_distance, actual_options in build_calls:
        expected_options: dict[str, Any] = {"build_pulsations": not subtract_temperature, **explicit_options}
        if kind == "binary":
            expected_options["components_distance"] = built_distance
        assert actual_options == expected_options
    for pulse_distance, actual_pulse_options in pulsation_calls:
        expected_pulse_options: dict[str, Any] = {}
        if kind == "binary":
            expected_pulse_options["components_distance"] = pulse_distance
            expected_pulse_options["component"] = "all"
        assert actual_pulse_options == expected_pulse_options


@pytest.mark.requires_ld
@pytest.mark.parametrize(
    ("selector_keyword", "selector_value", "selected_names"),
    [("component", "all", ("primary", "secondary")),
     ("component", "both", ("primary", "secondary")),
     ("component", "primary", ("primary",)),
     ("component", "secondary", ("secondary",)),
     ("components", "primary", ("primary", "secondary"))],
)
def test_binary_build_selector_controls_real_surface(
    selector_keyword: str,
    selector_value: Literal["all", "both", "primary", "secondary"],
    selected_names: tuple[str, ...],
) -> None:
    """Distinguish recognized aliases and single-star selection from an ignored keyword."""
    binary_system = cast("BinarySystem", make_system("binary", [MODE], secondary_modes=True))
    reference = binary_system.build_container(phase=0.2)
    fresh_container = type(reference).from_binary_system(binary_system, reference.position)
    fresh_container.set_time()
    if selector_keyword == "component":
        fresh_container.build(component=selector_value)
    else:
        fresh_container.build(components=selector_value)
    for component_name in ("primary", "secondary"):
        surface = getattr(fresh_container, component_name)
        if component_name in selected_names:
            assert surface.points.size > 0
            assert surface.faces.size > 0
            assert surface.temperatures.size == len(surface.faces)
            assert np.isfinite(surface.temperatures).all()
            assert surface.pulsations[0].point_harmonics is not None
        else:
            assert surface.points is None or surface.points.size == 0
            assert surface.temperatures is None or surface.temperatures.size == 0
            assert surface.pulsations[0].point_harmonics is None
    if len(selected_names) == 2:
        assert_complete_state_equal(fresh_container, reference)


@pytest.mark.requires_ld
@pytest.mark.parametrize("surface_phase", [0.2, 0.31])
def test_single_position_phase_drives_harmonics(
    surface_phase: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ignored build keyword cannot replace the position's physical phase or time."""
    from elisa.single_system.surface import pulsations as single_pulsations

    single_system = cast("SingleSystem", make_system("single", [MODE], spotted=True))
    actual_container = single_system.build_container(phase=surface_phase, build_pulsations=False)
    legacy_container = actual_container.copy()
    generated_phases: list[tuple[float, float]] = []
    original_harmonics = single_pulsations.generate_harmonics

    def record_harmonics(
        harmonic_star: StarContainer, com_x: float, phase: float, time: float,
    ) -> StarContainer:
        generated_phases.append((phase, time))
        return original_harmonics(harmonic_star, com_x=com_x, phase=phase, time=time)

    monkeypatch.setattr(single_pulsations, "generate_harmonics", record_harmonics)
    legacy_container.build(phase=0.73)
    plot.build_surface_with_temperature_reference(actual_container, "temperature", subtract_equilibrium=True)
    expected_time = 86400 * single_system.rotation_period * surface_phase
    assert generated_phases == [(surface_phase, expected_time), (surface_phase, expected_time)]
    mode_state = actual_container.star.pulsations[0]
    assert not mode_state.tidally_locked
    assert np.isnan(actual_container.star.synchronicity)
    assert_array_equal(mode_state.tilt_phi, mode_state.mode_axis_phi + 2 * np.pi * surface_phase)
    assert_complete_state_equal(actual_container, legacy_container)
