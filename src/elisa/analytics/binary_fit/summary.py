from __future__ import annotations

from multiprocessing.pool import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from elisa import settings
from elisa import umpy as up
from elisa import units as u
from elisa.analytics.binary_fit import io_tools
from elisa.analytics.binary_fit.shared import check_for_boundary_surface_potentials
from elisa.analytics.models import lc as lc_model
from elisa.analytics.models import serializers
from elisa.analytics.params import parameters
from elisa.analytics.params.parameters import BinaryInitialParameters
from elisa.base.error import MaxIterationError, MorphologyError
from elisa.binary_system.curves.community import RadialVelocitySystem
from elisa.binary_system.surface.gravity import calculate_polar_gravity_acceleration
from elisa.logger import getLogger
from elisa.utils import split_to_batches

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from elisa.analytics.binary_fit.lc_fit import LCFitLeastSquares, LCFitMCMC
    from elisa.analytics.binary_fit.rv_fit import RVFitLeastSquares, RVFitMCMC
    from elisa.types import Float


logger = getLogger("analytics.binary_fit.summary")

DASH_N = 126


def fit_lc_summary_with_error_propagation(  # noqa: C901, PLR0912, PLR0915
    fit_instance: LCFitMCMC | RVFitMCMC,
    path: str | None,
    percentiles: list[int],
    *,
    dimensionless_radii: bool = True,
) -> None:
    """Propagate errors of fitted parameters during MCMC fit by evaluating chain.

    This function propagates errors from the MCMC chain to derived binary
    parameters and generates a comprehensive summary report. The error
    propagation evaluates the full set of binary parameters across the
    MCMC chain and computes percentile-based confidence intervals.

    :param fit_instance: MCMC light curve fit instance with chain data.
    :type fit_instance: LCFitMCMC
    :param path: Path to store summary. If None, output goes to stdout.
    :type path: str | None
    :param percentiles: List of percentiles [lower, center, upper] for
        confidence intervals, e.g., [16, 50, 84] for 1-sigma.
    :type percentiles: list[int]
    :param dimensionless_radii: If True, radii provided in SMA units,
        otherwise in solar radii. Default is True.
    :type dimensionless_radii: bool
    :return: None. Summary is written to file or stdout.
    :rtype: None
    :raises ValueError: If MCMC chain is not found in fit_instance.
    """

    def propagate_errors(radius: NDArray, factor: NDArray) -> NDArray:
        """Propagate errors through radius scaling.

        :param radius: Input radius array with base value and uncertainties.
        :type radius: NDArray
        :param factor: Scaling factor with base value and uncertainties.
        :type factor: NDArray
        :return: Propagated error radius array.
        :rtype: NDArray
        """
        rel_rad = radius[1:] / radius[0]
        rel_factor = factor[1:] / factor[0]
        retval = np.empty(radius.shape)
        retval[0] = radius[0] * factor[0]
        retval[1:] = retval[0] * (rel_factor + rel_rad)
        return retval

    f: Any = None
    if path is not None:
        f = Path(path).open("w")  # noqa: SIM115
        write_fn: Callable[..., Any] = f.write
        line_sep: str = "\n"
    else:
        write_fn = print
        line_sep = ""

    flat_chain = fit_instance.flat_chain
    variable_labels = fit_instance.variable_labels
    normalization = fit_instance.normalization

    flat_params = fit_instance.flat_result
    params = fit_instance.result

    params_instance = BinaryInitialParameters(**params)
    fit_instance.fit_method_instance.fixed = params_instance.get_fixed(jsonify=False)
    fit_instance.fit_method_instance.constrained = params_instance.get_constrained(jsonify=False)
    fit_instance.fit_method_instance.fitable = params_instance.get_fitable(jsonify=False)

    error_msg = "MCMC chain was not found."
    if flat_chain is None:
        raise ValueError(error_msg)

    renormalized_chain = np.empty(flat_chain.shape)
    for ii, lbl in enumerate(variable_labels):
        renormalized_chain[:, ii] = parameters.renormalize_value(
            flat_chain[:, ii],
            normalization[lbl][0],
            normalization[lbl][1],
        )
    stop_idx: dict[str, int] = {}
    # creating param list of the binary system
    complete_param_list: list[str] = [
        "system@mass_ratio",
        "system@semi_major_axis",
        "system@asini",
        "system@inclination",
        "system@eccentricity",
        "system@argument_of_periastron",
        "system@gamma",
        "system@period",
        "system@primary_minimum_time",
        "system@additional_light",
        "system@phase_shift",
    ]

    component_param_list: list[str] = [
        "mass",
        "surface_potential",
        "synchronicity",
        "equivalent_radius",
        "polar_radius",
        "backward_radius",
        "side_radius",
        "forward_radius",
        "t_eff",
        "gravity_darkening",
        "albedo",
        "metallicity",
        "critical_potential",
        "polar_log_g",
        "bolometric_luminosity",
    ]

    stop_idx["system"] = len(complete_param_list)
    spot_numbers: dict[str, int] = {}
    pulsation_numbers: dict[str, int] = {}
    spot_lbls: dict[str, set[str]] = {}
    pulse_lbls: dict[str, set[str]] = {}

    for component in settings.BINARY_COUNTERPARTS:
        complete_param_list += [f"{component}@{prm}" for prm in component_param_list]

        stop_idx[f"{component}_params"] = len(complete_param_list)
        spot_lbls[component] = {lbl.split("@")[2] for lbl in flat_params if f"{component}@spot" in lbl}
        spot_numbers[component] = len(spot_lbls[component])
        for spot in spot_lbls[component]:
            complete_param_list += [
                f"{component}@spot@{spot}@longitude",
                f"{component}@spot@{spot}@latitude",
                f"{component}@spot@{spot}@angular_radius",
                f"{component}@spot@{spot}@temperature_factor",
            ]
        stop_idx[f"{component}_spots"] = len(complete_param_list)
        pulse_lbls[component] = {lbl.split("@")[2] for lbl in flat_params if f"{component}@pulsation" in lbl}
        pulsation_numbers[component] = len(pulse_lbls[component])
        for mode in pulse_lbls[component]:
            complete_param_list += [
                f"{component}@pulsation@{mode}@l",
                f"{component}@pulsation@{mode}@m",
                f"{component}@pulsation@{mode}@amplitude",
                f"{component}@pulsation@{mode}@frequency",
                f"{component}@pulsation@{mode}@start_phase",
                f"{component}@pulsation@{mode}@mode_axis_phi",
                f"{component}@pulsation@{mode}@mode_axis_theta",
            ]
        stop_idx[f"{component}_pulsations"] = len(complete_param_list)

    param_columns: dict[str, int] = {lbl: ii for ii, lbl in enumerate(complete_param_list)}

    # obtaining binary parameters for each item in MCMC chain
    args: tuple[Any, ...] = (
        fit_instance,
        param_columns,
        stop_idx,
        spot_numbers,
        pulsation_numbers,
        len(component_param_list),
    )
    full_chain: NDArray = _manage_chain_evaluation(renormalized_chain, evaluate_binary_params, *args)

    full_chain_mask: NDArray = (~np.isnan(full_chain)).any(axis=1)
    full_chain = full_chain[full_chain_mask]

    # avoiding np warnings about NANs
    calculated_percentiles: NDArray = np.empty((3, full_chain.shape[1]))
    calculated_percentiles[:] = up.NaN
    full_chain_mask = (~np.isnan(full_chain)).any(axis=0)

    # evaluating posterior distribution of each binary parameter
    calculated_percentiles[:, full_chain_mask] = np.percentile(full_chain[:, full_chain_mask], percentiles, axis=0)
    full_chain_results: NDArray = np.vstack(
        (
            calculated_percentiles[1, :],
            calculated_percentiles[1, :] - calculated_percentiles[0, :],
            calculated_percentiles[2, :] - calculated_percentiles[1, :],
        ),
    )

    # output to screen/file
    intro: tuple[Callable[..., Any], str, str, str, str, str, str, str] = (
        write_fn,
        "Parameter",
        "value",
        "-1 sigma",
        "+1 sigma",
        "unit",
        "status",
        line_sep,
    )
    write_fn(f"\nBINARY SYSTEM{line_sep}")
    io_tools.write_ln(*intro)
    write_fn(f"{'-' * DASH_N}{line_sep}")

    io_tools.write_propagated_ln(
        full_chain_results[:, param_columns["system@mass_ratio"]],
        flat_params,
        "system@mass_ratio",
        "Mass ratio (q=M_2/M_1):",
        write_fn,
        line_sep,
        "-",
    )

    sma_value: NDArray = full_chain_results[:, param_columns["system@semi_major_axis"]]
    sma: NDArray = (
        (sma_value * u.DefaultBinarySystemUnits.system.semi_major_axis)
        .to(u.DefaultBinarySystemInputUnits.system.semi_major_axis)
        .value
    )
    sma_factor: NDArray = sma if not dimensionless_radii else np.array([1.0, 0.0, 0.0])
    sma_unit: str = "solRad" if not dimensionless_radii else "SMA"
    io_tools.write_propagated_ln(
        sma,
        flat_params,
        "system@semi_major_axis",
        "Semi major axis (a):",
        write_fn,
        line_sep,
        "solRad",
    )

    asini_value = full_chain_results[:, param_columns["system@asini"]]
    asini = (
        (asini_value * u.DefaultBinarySystemUnits.system.semi_major_axis)
        .to(u.DefaultBinarySystemInputUnits.system.semi_major_axis)
        .value
    )
    io_tools.write_propagated_ln(
        asini,
        flat_params,
        "system@asini",
        "a*sin(i):",
        write_fn,
        line_sep,
        "solRad",
    )

    incl = (
        (full_chain_results[:, param_columns["system@inclination"]] * u.DefaultBinarySystemUnits.system.inclination)
        .to(u.deg)
        .value
    )
    io_tools.write_propagated_ln(
        incl,
        flat_params,
        "system@inclination",
        "Inclination (i):",
        write_fn,
        line_sep,
        "deg",
    )

    io_tools.write_propagated_ln(
        full_chain_results[:, param_columns["system@eccentricity"]],
        flat_params,
        "system@eccentricity",
        "Eccentricity (e):",
        write_fn,
        line_sep,
        "-",
    )

    omega = (
        (
            full_chain_results[:, param_columns["system@argument_of_periastron"]]
            * u.DefaultBinarySystemUnits.system.argument_of_periastron
        )
        .to(u.deg)
        .value
    )
    io_tools.write_propagated_ln(
        omega,
        flat_params,
        "system@argument_of_periastron",
        "Argument of periastron (omega):",
        write_fn,
        line_sep,
        "deg",
    )

    gamma = (
        (full_chain_results[:, param_columns["system@gamma"]] * u.DefaultBinarySystemUnits.system.gamma)
        .to(u.km / u.s)
        .value
    )
    io_tools.write_propagated_ln(
        gamma,
        flat_params,
        "system@gamma",
        "Centre of mass velocity (gamma):",
        write_fn,
        line_sep,
        "km/s",
    )

    io_tools.write_propagated_ln(
        full_chain_results[:, param_columns["system@period"]],
        flat_params,
        "system@period",
        "Orbital period (P):",
        write_fn,
        line_sep,
        "d",
    )

    if "system@primary_minimum_time" in flat_params:
        io_tools.write_param_ln(
            flat_params,
            "system@primary_minimum_time",
            "Time of primary minimum (T0):",
            write_fn,
            line_sep,
        )

    io_tools.write_propagated_ln(
        full_chain_results[:, param_columns["system@additional_light"]],
        flat_params,
        "system@additional_light",
        "Additional light (l_3):",
        write_fn,
        line_sep,
        "-",
    )

    io_tools.write_propagated_ln(
        full_chain_results[:, param_columns["system@phase_shift"]],
        flat_params,
        "system@phase_shift",
        "Phase shift:",
        write_fn,
        line_sep,
        "-",
    )

    write_fn(f"{'-' * DASH_N}{line_sep}")

    for component in settings.BINARY_COUNTERPARTS:
        comp_n = 1 if component == "primary" else 2
        write_fn(f"{component.upper()} COMPONENT{line_sep}")
        io_tools.write_ln(*intro)
        write_fn(f"{'-' * DASH_N}{line_sep}")

        mass = (
            (full_chain_results[:, param_columns[f"{component}@mass"]] * u.DefaultBinarySystemUnits.component.mass)
            .to(u.solMass)
            .value
        )
        io_tools.write_propagated_ln(
            mass,
            flat_params,
            f"{component}@mass",
            f"Mass (M_{comp_n}):",
            write_fn,
            line_sep,
            "solMas",
        )

        io_tools.write_propagated_ln(
            full_chain_results[:, param_columns[f"{component}@surface_potential"]],
            flat_params,
            f"{component}@surface_potential",
            f"Surface potential (Omega_{comp_n}):",
            write_fn,
            line_sep,
            "-",
        )
        io_tools.write_propagated_ln(
            full_chain_results[:, param_columns[f"{component}@critical_potential"]],
            flat_params,
            f"{component}@critical_potential",
            "Critical potential at L_1:",
            write_fn,
            line_sep,
            "-",
        )
        io_tools.write_propagated_ln(
            full_chain_results[:, param_columns[f"{component}@synchronicity"]],
            flat_params,
            f"{component}@synchronicity",
            f"Synchronicity (F_{comp_n}):",
            write_fn,
            line_sep,
            "-",
        )
        io_tools.write_propagated_ln(
            full_chain_results[:, param_columns[f"{component}@polar_log_g"]],
            flat_params,
            f"{component}@polar_log_g",
            "Polar gravity (log g):",
            write_fn,
            line_sep,
            "log(cgs)",
        )

        r_equiv = propagate_errors(
            full_chain_results[:, param_columns[f"{component}@equivalent_radius"]],
            sma_factor,
        )
        r_polar = propagate_errors(
            full_chain_results[:, param_columns[f"{component}@polar_radius"]],
            sma_factor,
        )
        r_backw = propagate_errors(
            full_chain_results[:, param_columns[f"{component}@backward_radius"]],
            sma_factor,
        )
        r_side = propagate_errors(
            full_chain_results[:, param_columns[f"{component}@side_radius"]],
            sma_factor,
        )

        io_tools.write_propagated_ln(
            r_equiv,
            flat_params,
            f"{component}@equivalent_radius",
            "Equivalent radius (R_equiv):",
            write_fn,
            line_sep,
            sma_unit,
        )
        write_fn(f"\nPeriastron radii{line_sep}")

        io_tools.write_propagated_ln(
            r_polar,
            flat_params,
            f"{component}@polar_radius",
            "Polar radius:",
            write_fn,
            line_sep,
            sma_unit,
        )
        io_tools.write_propagated_ln(
            r_backw,
            flat_params,
            f"{component}@backw_radius",
            "Backward radius:",
            write_fn,
            line_sep,
            sma_unit,
        )
        io_tools.write_propagated_ln(
            r_side,
            flat_params,
            f"{component}@side_radius",
            "Side radius:",
            write_fn,
            line_sep,
            sma_unit,
        )
        if fit_instance.morphology != "over-contact":
            r_forw = propagate_errors(
                full_chain_results[:, param_columns[f"{component}@forward_radius"]],
                sma_factor,
            )
            io_tools.write_propagated_ln(
                r_forw,
                flat_params,
                f"{component}@forward_radius",
                "Forward radius:",
                write_fn,
                line_sep,
                sma_unit,
            )

        write_fn(f"\nAtmospheric parameters{line_sep}")
        io_tools.write_propagated_ln(
            full_chain_results[:, param_columns[f"{component}@t_eff"]],
            flat_params,
            f"{component}@t_eff",
            f"Effective temperature (T_eff{comp_n}):",
            write_fn,
            line_sep,
            "K",
        )

        l_bol = (
            (full_chain_results[:, param_columns[f"{component}@bolometric_luminosity"]] * u.LUMINOSITY_UNIT)
            .to("L_sun")
            .value
        )
        io_tools.write_propagated_ln(
            l_bol,
            flat_params,
            f"{component}@bolometric_luminosity",
            "Bolometric luminosity (L_bol): ",
            write_fn,
            line_sep,
            "L_sol",
        )
        io_tools.write_propagated_ln(
            full_chain_results[:, param_columns[f"{component}@gravity_darkening"]],
            flat_params,
            f"{component}@gravity_darkening",
            f"Gravity darkening factor (G_{comp_n}):",
            write_fn,
            line_sep,
            "-",
        )
        io_tools.write_propagated_ln(
            full_chain_results[:, param_columns[f"{component}@albedo"]],
            flat_params,
            f"{component}@albedo",
            f"Albedo (A_{comp_n}):",
            write_fn,
            line_sep,
            "-",
        )
        io_tools.write_propagated_ln(
            full_chain_results[:, param_columns[f"{component}@metallicity"]],
            flat_params,
            f"{component}@metallicity",
            "Metallicity (log10(X_Fe/X_H)):",
            write_fn,
            line_sep,
            "-",
        )

        if spot_numbers[component] > 0:
            write_fn(f"{'-' * DASH_N}{line_sep}")
            write_fn(f"{component.upper()} SPOTS{line_sep}")
            io_tools.write_ln(*intro)

            for spot_lbl in spot_lbls[component]:
                longitude = (
                    (
                        full_chain_results[:, param_columns[f"{component}@spot@{spot_lbl}@longitude"]]
                        * u.DefaultSpotUnits.longitude
                    )
                    .to(u.deg)
                    .value
                )
                io_tools.write_propagated_ln(
                    longitude,
                    flat_params,
                    f"{component}@spot@{spot_lbl}@longitude",
                    "Longitude: ",
                    write_fn,
                    line_sep,
                    "deg",
                )
                latitude = (
                    (
                        full_chain_results[:, param_columns[f"{component}@spot@{spot_lbl}@latitude"]]
                        * u.DefaultSpotUnits.latitude
                    )
                    .to(u.deg)
                    .value
                )
                io_tools.write_propagated_ln(
                    latitude,
                    flat_params,
                    f"{component}@spot@{spot_lbl}@latitude",
                    "Latitude: ",
                    write_fn,
                    line_sep,
                    "deg",
                )
                ang_rad = (
                    (
                        full_chain_results[:, param_columns[f"{component}@spot@{spot_lbl}@angular_radius"]]
                        * u.DefaultSpotUnits.angular_radius
                    )
                    .to(u.deg)
                    .value
                )
                io_tools.write_propagated_ln(
                    ang_rad,
                    flat_params,
                    f"{component}@spot@{spot_lbl}@angular_radius",
                    "Angular radius: ",
                    write_fn,
                    line_sep,
                    "deg",
                )
                io_tools.write_propagated_ln(
                    full_chain_results[:, param_columns[f"{component}@spot@{spot_lbl}@temperature_factor"]],
                    flat_params,
                    f"{component}@spot@{spot_lbl}@temperature_factor",
                    "Temperature factor (T_spot/T_eff): ",
                    write_fn,
                    line_sep,
                    "-",
                )

        if pulsation_numbers[component] > 0:
            write_fn(f"{'-' * DASH_N}{line_sep}")
            write_fn(f"{component.upper()} PULSATIONS{line_sep}")
            io_tools.write_ln(*intro)

            for pulse_lbl in pulse_lbls[component]:
                io_tools.write_propagated_ln(
                    full_chain_results[:, param_columns[f"{component}@pulsation@{pulse_lbl}@l"]],
                    flat_params,
                    f"{component}@pulsation@{pulse_lbl}@l",
                    "Angular degree (l): ",
                    write_fn,
                    line_sep,
                    "-",
                )
                io_tools.write_propagated_ln(
                    full_chain_results[:, param_columns[f"{component}@pulsation@{pulse_lbl}@m"]],
                    flat_params,
                    f"{component}@pulsation@{pulse_lbl}@m",
                    "Azimuthal order (m):  ",
                    write_fn,
                    line_sep,
                    "-",
                )
                io_tools.write_propagated_ln(
                    full_chain_results[:, param_columns[f"{component}@pulsation@{pulse_lbl}@amplitude"]],
                    flat_params,
                    f"{component}@pulsation@{pulse_lbl}@m",
                    "Amplitude (A): ",
                    write_fn,
                    line_sep,
                    "-",
                )
                io_tools.write_propagated_ln(
                    full_chain_results[:, param_columns[f"{component}@pulsation@{pulse_lbl}@frequency"]],
                    flat_params,
                    f"{component}@pulsation@{pulse_lbl}@frequency",
                    "Frequency (f): ",
                    write_fn,
                    line_sep,
                    "c/d",
                )
                io_tools.write_propagated_ln(
                    full_chain_results[:, param_columns[f"{component}@pulsation@{pulse_lbl}@start_phase"]],
                    flat_params,
                    f"{component}@pulsation@{pulse_lbl}@start_phase",
                    "Initial phase (at T_0): ",
                    write_fn,
                    line_sep,
                    "-",
                )
                mode_phi = (
                    (
                        full_chain_results[:, param_columns[f"{component}@spot@{pulse_lbl}@mode_axis_phi"]]
                        * u.DefaultPulsationsUnits.mode_axis_phi
                    )
                    .to(u.deg)
                    .value
                )
                io_tools.write_propagated_ln(
                    mode_phi,
                    flat_params,
                    f"{component}@pulsation@{pulse_lbl}@mode_axis_phi",
                    "Longitude of mode axis: ",
                    write_fn,
                    line_sep,
                    "deg",
                )
                mode_theta = (
                    (
                        full_chain_results[:, param_columns[f"{component}@spot@{pulse_lbl}@mode_axis_theta"]]
                        * u.DefaultPulsationsUnits.mode_axis_theta
                    )
                    .to(u.deg)
                    .value
                )
                io_tools.write_propagated_ln(
                    mode_theta,
                    flat_params,
                    f"{component}@pulsation@{pulse_lbl}@mode_axis_theta",
                    "Latitude of mode axis: ",
                    write_fn,
                    line_sep,
                    "deg",
                )

        write_fn(f"{'-' * DASH_N}{line_sep}")

    if f is not None:
        f.close()


def evaluate_binary_params(*args: Any) -> NDArray:
    """Evaluate MCMC (sub)chain for error propagation in fit summary.

    Evaluates binary system parameters across the MCMC chain by constructing
    binary instances from normalized chain values and extracting parameter
    values, derived quantities, and spot/pulsation properties.

    :param args: Tuple containing (fit_instance, param_columns, stop_idx,
        spot_numbers, pulsation_numbers, component_param_length,
        renormalized_chain). See source for detailed parameter descriptions.
    :type args: Any
    :return: Array of evaluated parameters with shape
        (n_chain_samples, n_parameters). Contains NaN for failed evaluations.
    :rtype: NDArray
    """
    fit_instance: Any
    param_columns: dict[str, int]
    stop_idx: dict[str, int]
    spot_numbers: dict[str, int]
    pulsation_numbers: dict[str, int]
    cpl: int
    renormalized_chain: NDArray

    fit_instance, param_columns, stop_idx, spot_numbers, pulsation_numbers, cpl, renormalized_chain = args
    full_chain: NDArray = np.empty((renormalized_chain.shape[0], len(param_columns)))
    full_chain[:] = np.nan

    for ii in tqdm(range(renormalized_chain.shape[0])):
        init_binary_kwargs: dict[str, Any] = parameters.prepare_properties_set(
            renormalized_chain[ii, :],
            fit_instance.fit_method_instance.fitable.keys(),
            fit_instance.fit_method_instance.constrained,
            fit_instance.fit_method_instance.fixed,
        )

        try:
            binary_instance: Any = lc_model.prepare_binary(_verify=False, **init_binary_kwargs)
            for var_label in list(param_columns.keys())[: stop_idx["system"]]:
                full_chain[ii, param_columns[var_label]] = getattr(binary_instance, var_label.split("@")[1], None)

            for component in settings.BINARY_COUNTERPARTS:
                star_instance: Any = getattr(binary_instance, component)
                for var_label in list(param_columns.keys())[
                    stop_idx[f"{component}_params"] - cpl : stop_idx[f"{component}_params"] - 3
                ]:
                    full_chain[ii, param_columns[var_label]] = getattr(star_instance, var_label.split("@")[1], None)

                polar_g: Float = (
                    calculate_polar_gravity_acceleration(
                        star_instance,
                        1 - binary_instance.eccentricity,
                        binary_instance.mass_ratio,
                        component=component,
                        semi_major_axis=binary_instance.semi_major_axis,
                        synchronicity=star_instance.synchronicity,
                        logg=True,
                    )
                    + 2
                )
                full_chain[ii, param_columns[f"{component}@polar_log_g"]] = polar_g

                crit_pot: Float = binary_instance.critical_potential(component, 1.0 - binary_instance.eccentricity)
                full_chain[ii, param_columns[f"{component}@critical_potential"]] = crit_pot

                l_bol: Float = binary_instance.calculate_bolometric_luminosity(components=component)[component]
                full_chain[ii, param_columns[f"{component}@bolometric_luminosity"]] = l_bol

                if spot_numbers[component] == 0 and pulsation_numbers[component] == 0:
                    continue

                ref_idx: int = stop_idx[f"{component}_params"]
                for spot_idx in range(spot_numbers[component]):
                    full_chain[ii, ref_idx + 4 * spot_idx] = star_instance.spots[spot_idx].longitude
                    full_chain[ii, ref_idx + 4 * spot_idx + 1] = star_instance.spots[spot_idx].latitude
                    full_chain[ii, ref_idx + 4 * spot_idx + 2] = star_instance.spots[spot_idx].angular_radius
                    full_chain[ii, ref_idx + 4 * spot_idx + 3] = star_instance.spots[spot_idx].temperature_factor

                ref_idx = stop_idx[f"{component}_spots"]
                for pulse_idx in range(pulsation_numbers[component]):
                    full_chain[ii, ref_idx + 7 * pulse_idx] = star_instance.pulsations[pulse_idx].l
                    full_chain[ii, ref_idx + 7 * pulse_idx + 1] = star_instance.pulsations[pulse_idx].m
                    full_chain[ii, ref_idx + 7 * pulse_idx + 2] = star_instance.pulsations[pulse_idx].amplitude
                    full_chain[ii, ref_idx + 7 * pulse_idx + 3] = star_instance.pulsations[pulse_idx].frequency
                    full_chain[ii, ref_idx + 7 * pulse_idx + 4] = star_instance.pulsations[pulse_idx].start_phase
                    full_chain[ii, ref_idx + 7 * pulse_idx + 5] = star_instance.pulsations[pulse_idx].mode_axis_phi
                    full_chain[ii, ref_idx + 7 * pulse_idx + 6] = star_instance.pulsations[pulse_idx].mode_axis_theta

        except (MaxIterationError, MorphologyError):
            continue

    return full_chain


def simple_lc_fit_summary(  # noqa: C901, PLR0912, PLR0915
    fit_instance: LCFitLeastSquares | LCFitMCMC | RVFitMCMC,
    path: str | None,
    *,
    dimensionless_radii: bool = True,
) -> None:
    """Generate detailed light curve fit summary report.

    Produces comprehensive summary of fitting results with all fitted or
    derived binary parameters. Summary includes component parameters,
    surface properties, atmospheric parameters, and optionally spots and
    pulsations.

    :param fit_instance: Light curve fit instance with results.
    :type fit_instance: LCFitLeastSquares | LCFitMCMC
    :param path: Path to store summary. If None, output goes to stdout.
    :type path: str | None
    :param dimensionless_radii: If True, radii in SMA units (default),
        otherwise in solar radii.
    :type dimensionless_radii: bool
    :return: None. Summary written to file or stdout.
    :rtype: None
    """
    f: Any = None
    if path is not None:
        f = Path(path).open("w")  # noqa: SIM115
        write_fn: Callable[..., Any] = f.write
        line_sep: str = "\n"
    else:
        write_fn = print
        line_sep = ""

    try:
        result_dict: dict[str, Any] = fit_instance.flat_result
        result_dict = check_for_boundary_surface_potentials(result_dict, fit_instance.morphology)
        b_kwargs: dict[str, Any] = {key: val["value"] for key, val in result_dict.items()}
        binary_instance: Any = lc_model.prepare_binary(_verify=False, **b_kwargs)

        intro: tuple[Callable[..., Any], str, str, str, str, str, str, str] = (
            write_fn,
            "Parameter",
            "value",
            "-1 sigma",
            "+1 sigma",
            "unit",
            "status",
            line_sep,
        )
        write_fn(f"\nBINARY SYSTEM{line_sep}")
        io_tools.write_ln(*intro)
        write_fn(f"{'-' * DASH_N}{line_sep}")

        q_desig: str = "Mass ratio (q=M_2/M_1):"
        a_desig: str = "Semi major axis (a):"

        if "system@mass_ratio" in result_dict:
            io_tools.write_param_ln(
                result_dict,
                "system@mass_ratio",
                q_desig,
                write_fn,
                line_sep,
                3,
            )
            io_tools.write_param_ln(
                result_dict,
                "system@semi_major_axis",
                a_desig,
                write_fn,
                line_sep,
                3,
            )
            sma_factor: Float = result_dict["system@semi_major_axis"]["value"] if not dimensionless_radii else 1.0
            sma_unit: str = str(result_dict["system@semi_major_axis"]["unit"]) if not dimensionless_radii else "SMA"
        else:
            io_tools.write_ln(write_fn, q_desig, binary_instance.mass_ratio, "", "", "", "derived", line_sep, 3)
            sma: Float = (
                (binary_instance.semi_major_axis * u.DefaultBinarySystemUnits.system.semi_major_axis)
                .to(u.DefaultBinarySystemInputUnits.system.semi_major_axis)
                .value
            )
            io_tools.write_ln(write_fn, a_desig, sma, "", "", "solRad", "derived", line_sep, 3)
            sma_factor = sma
            sma_unit = "solRad" if not dimensionless_radii else "SMA"

        io_tools.write_param_ln(result_dict, "system@inclination", "Inclination (i):", write_fn, line_sep, 2)
        io_tools.write_param_ln(result_dict, "system@eccentricity", "Eccentricity (e):", write_fn, line_sep, 2)
        io_tools.write_param_ln(
            result_dict,
            "system@argument_of_periastron",
            "Argument of periastron (omega):",
            write_fn,
            line_sep,
            2,
        )
        io_tools.write_param_ln(result_dict, "system@period", "Orbital period (P):", write_fn, line_sep)

        if "system@primary_minimum_time" in result_dict:
            io_tools.write_param_ln(
                result_dict,
                "system@primary_minimum_time",
                "Time of primary minimum (T0):",
                write_fn,
                line_sep,
            )
        if "system@additional_light" in result_dict:
            io_tools.write_param_ln(
                result_dict,
                "system@additional_light",
                "Additional light (l_3):",
                write_fn,
                line_sep,
                4,
            )

        p_desig = "Phase shift:"
        if "system@phase_shift" in result_dict:
            io_tools.write_param_ln(
                result_dict,
                "system@phase_shift",
                p_desig,
                write_fn,
                line_sep,
            )
        else:
            io_tools.write_ln(
                write_fn,
                p_desig,
                0.0,
                "-",
                "-",
                "-",
                "Fixed",
                line_sep,
            )

        write_fn(f"{'-' * DASH_N}{line_sep}")

        for component in settings.BINARY_COUNTERPARTS:
            comp_n: int = 1 if component == "primary" else 2
            star_instance: Any = getattr(binary_instance, component)
            write_fn(f"{component.upper()} COMPONENT{line_sep}")
            io_tools.write_ln(*intro)
            write_fn(f"{'-' * DASH_N}{line_sep}")

            m_desig: str = f"Mass (M_{comp_n}):"
            if f"{component}@mass" in result_dict:
                io_tools.write_param_ln(
                    result_dict,
                    f"{component}@mass",
                    m_desig,
                    write_fn,
                    line_sep,
                    3,
                )
            else:
                mass: Float = (star_instance.mass * u.DefaultBinarySystemUnits.component.mass).to(u.solMass).value
                io_tools.write_ln(
                    write_fn,
                    m_desig,
                    mass,
                    "-",
                    "-",
                    "solMass",
                    "Derived",
                    line_sep,
                    3,
                )

            io_tools.write_param_ln(
                result_dict,
                f"{component}@surface_potential",
                f"Surface potential (Omega_{comp_n}):",
                write_fn,
                line_sep,
                4,
            )

            crit_pot: Float = binary_instance.critical_potential(component, 1.0 - binary_instance.eccentricity)
            io_tools.write_ln(
                write_fn,
                "Critical potential at L_1:",
                crit_pot,
                "-",
                "-",
                "-",
                "Derived",
                line_sep,
                4,
            )

            f_desig: str = f"Synchronicity (F_{comp_n}):"

            if f"{component}@synchronicity" in result_dict:
                io_tools.write_param_ln(
                    result_dict,
                    f"{component}@synchronicity",
                    f_desig,
                    write_fn,
                    line_sep,
                    3,
                )
            else:
                io_tools.write_ln(
                    write_fn,
                    f_desig,
                    star_instance.synchronicity,
                    "-",
                    "-",
                    "-",
                    "Fixed",
                    line_sep,
                    3,
                )

            polar_g: Float = (
                calculate_polar_gravity_acceleration(
                    star_instance,
                    1 - binary_instance.eccentricity,
                    binary_instance.mass_ratio,
                    component=component,
                    semi_major_axis=binary_instance.semi_major_axis,
                    synchronicity=star_instance.synchronicity,
                    logg=True,
                )
                + 2
            )

            io_tools.write_ln(
                write_fn,
                "Polar gravity (log g):",
                polar_g,
                "-",
                "-",
                "log(cgs)",
                "Derived",
                line_sep,
                3,
            )

            io_tools.write_ln(
                write_fn,
                "Equivalent radius (R_equiv):",
                star_instance.equivalent_radius * sma_factor,
                "-",
                "-",
                sma_unit,
                "Derived",
                line_sep,
                5,
            )

            write_fn(f"\nPeriastron radii{line_sep}")

            io_tools.write_ln(
                write_fn,
                "Polar radius:",
                star_instance.polar_radius * sma_factor,
                "-",
                "-",
                sma_unit,
                "Derived",
                line_sep,
                5,
            )
            io_tools.write_ln(
                write_fn,
                "Backward radius:",
                star_instance.backward_radius * sma_factor,
                "-",
                "-",
                sma_unit,
                "Derived",
                line_sep,
                5,
            )
            io_tools.write_ln(
                write_fn,
                "Side radius:",
                star_instance.side_radius * sma_factor,
                "-",
                "-",
                sma_unit,
                "Derived",
                line_sep,
                5,
            )

            if getattr(star_instance, "forward_radius", None) is not None:
                io_tools.write_ln(
                    write_fn,
                    "Forward radius:",
                    star_instance.forward_radius * sma_factor,
                    "-",
                    "-",
                    sma_unit,
                    "Derived",
                    line_sep,
                    5,
                )

            write_fn(f"\nAtmospheric parameters{line_sep}")

            io_tools.write_param_ln(
                result_dict,
                f"{component}@t_eff",
                f"Effective temperature (T_eff{comp_n}):",
                write_fn,
                line_sep,
                0,
            )

            l_bol: Float = (
                (binary_instance.calculate_bolometric_luminosity(components=component)[component] * u.LUMINOSITY_UNIT)
                .to("L_sun")
                .value
            )

            io_tools.write_ln(
                write_fn,
                "Bolometric luminosity (L_bol): ",
                l_bol,
                "-",
                "-",
                "L_Sol",
                "Derived",
                line_sep,
                2,
            )

            f_desig = f"Gravity darkening factor (G_{comp_n}):"
            if f"{component}@gravity_darkening" in result_dict:
                io_tools.write_param_ln(
                    result_dict,
                    f"{component}@gravity_darkening",
                    f_desig,
                    write_fn,
                    line_sep,
                    3,
                )
            else:
                args = (
                    write_fn,
                    f_desig,
                    star_instance.gravity_darkening,
                    "-",
                    "-",
                    "-",
                    "Derived",
                    line_sep,
                    3,
                )
                io_tools.write_ln(*args)

            f_desig = f"Albedo (A_{comp_n}):"
            if f"{component}@albedo" in result_dict:
                io_tools.write_param_ln(
                    result_dict,
                    f"{component}@albedo",
                    f_desig,
                    write_fn,
                    line_sep,
                    3,
                )
            else:
                args = (
                    write_fn,
                    f_desig,
                    star_instance.albedo,
                    "-",
                    "-",
                    "-",
                    "Derived",
                    line_sep,
                    3,
                )
                io_tools.write_ln(*args)

            met_desig = "Metallicity (log10(X_Fe/X_H)):"
            if f"{component}@metallicity" in result_dict:
                io_tools.write_param_ln(
                    result_dict,
                    f"{component}@metallicity",
                    met_desig,
                    write_fn,
                    line_sep,
                    2,
                )
            else:
                io_tools.write_ln(
                    write_fn,
                    met_desig,
                    star_instance.metallicity,
                    "-",
                    "-",
                    "-",
                    "Fixed",
                    line_sep,
                    2,
                )

            if star_instance.has_spots():
                write_fn(f"{'-' * DASH_N}{line_sep}")
                write_fn(f"{component.upper()} SPOTS{line_sep}")
                io_tools.write_ln(*intro)

                for spot in fit_instance.result[component]["spots"]:
                    spot: dict[str, Any]
                    write_fn(f"{'-' * DASH_N}{line_sep}")
                    write_fn(f"Spot: {spot['label']} {line_sep}")
                    write_fn(f"{'-' * DASH_N}{line_sep}")

                    io_tools.write_param_ln(
                        spot,
                        "longitude",
                        "Longitude: ",
                        write_fn,
                        line_sep,
                        3,
                    )
                    io_tools.write_param_ln(
                        spot,
                        "latitude",
                        "Latitude: ",
                        write_fn,
                        line_sep,
                        3,
                    )
                    io_tools.write_param_ln(
                        spot,
                        "angular_radius",
                        "Angular radius: ",
                        write_fn,
                        line_sep,
                        3,
                    )
                    io_tools.write_param_ln(
                        spot,
                        "temperature_factor",
                        "Temperature factor (T_spot/T_eff): ",
                        write_fn,
                        line_sep,
                        3,
                    )

            if star_instance.has_pulsations():
                write_fn(f"{'-' * DASH_N}{line_sep}")
                write_fn(f"{component.upper()} PULSATIONS{line_sep}")
                io_tools.write_ln(*intro)

                for mode in fit_instance.result[component]["pulsations"]:
                    mode: dict[str, Any]
                    write_fn(f"{'-' * DASH_N}{line_sep}")
                    write_fn(f"Pulsation: {mode['label']} {line_sep}")
                    write_fn(f"{'-' * DASH_N}{line_sep}")

                    io_tools.write_param_ln(
                        mode,
                        "l",
                        "Angular degree (l): ",
                        write_fn,
                        line_sep,
                        0,
                    )
                    io_tools.write_param_ln(
                        mode,
                        "m",
                        "Azimuthal order (m): ",
                        write_fn,
                        line_sep,
                        0,
                    )
                    io_tools.write_param_ln(
                        mode,
                        "amplitude",
                        "Amplitude (A): ",
                        write_fn,
                        line_sep,
                        0,
                    )
                    io_tools.write_param_ln(
                        mode,
                        "frequency",
                        "Frequency (f): ",
                        write_fn,
                        line_sep,
                        0,
                    )
                    io_tools.write_param_ln(
                        mode,
                        "start_phase",
                        "Initial phase (at T_0): ",
                        write_fn,
                        line_sep,
                        3,
                    )
                    io_tools.write_param_ln(
                        mode,
                        "mode_axis_phi",
                        "Longitude of mode axis: ",
                        write_fn,
                        line_sep,
                        1,
                    )
                    io_tools.write_param_ln(mode, "mode_axis_theta", "Latitude of mode axis: ", write_fn, line_sep, 1)

            write_fn(f"{'-' * DASH_N}{line_sep}")

        if result_dict.get("r_squared", False):
            if result_dict["r_squared"]["value"] is not None:
                io_tools.write_param_ln(result_dict, "r_squared", "Fit R^2: ", write_fn, line_sep, 6)

            write_fn(f"{'-' * DASH_N}{line_sep}")
    finally:
        if f is not None:
            f.close()


def simple_rv_fit_summary(
    fit_instance: LCFitLeastSquares | RVFitLeastSquares | LCFitMCMC | RVFitMCMC, path: str | None,
) -> None:
    """Generate detailed radial velocity fit summary report.

    Produces comprehensive summary of radial velocity fitting results with
    all fitted or derived binary parameters including masses, orbital
    parameters, and systemic velocity.

    :param fit_instance: Radial velocity fit instance with results.
    :type fit_instance: LCFitLeastSquares | LCFitMCMC
    :param path: Path to store summary. If None, output goes to stdout.
    :type path: str | None
    :return: None. Summary written to file or stdout.
    :rtype: None
    """
    f: Any = None
    if path is not None:
        f = Path(path).open("w")  # noqa: SIM115
        write_fn: Callable[..., Any] = f.write
        line_sep: str = "\n"
    else:
        write_fn = print
        line_sep = ""

    try:
        write_fn(f"\n{'-' * DASH_N}{line_sep}")
        io_tools.write_ln(
            write_fn,
            "Parameter",
            "value",
            "-1 sigma",
            "+1 sigma",
            "unit",
            "status",
            line_sep,
        )
        write_fn(f"{'-' * DASH_N}{line_sep}")
        result_dict: dict[str, Any] = fit_instance.flat_result

        if "system@mass_ratio" in result_dict:
            io_tools.write_param_ln(
                result_dict,
                "system@mass_ratio",
                "Mass ratio (q=M_2/M_1):",
                write_fn,
                line_sep,
                3,
            )
            io_tools.write_param_ln(
                result_dict,
                "system@asini",
                "a*sin(i):",
                write_fn,
                line_sep,
                2,
            )
        else:
            io_tools.write_param_ln(
                result_dict,
                "primary@mass",
                "Primary mass:",
                write_fn,
                line_sep,
                3,
            )
            io_tools.write_param_ln(
                result_dict,
                "secondary@mass",
                "Secondary mass:",
                write_fn,
                line_sep,
                3,
            )
            io_tools.write_param_ln(
                result_dict,
                "system@inclination",
                "Inclination(i):",
                write_fn,
                line_sep,
                3,
            )

        io_tools.write_param_ln(
            result_dict,
            "system@eccentricity",
            "Eccentricity (e):",
            write_fn,
            line_sep,
        )
        io_tools.write_param_ln(
            result_dict,
            "system@argument_of_periastron",
            "Argument of periastron (omega):",
            write_fn,
            line_sep,
        )
        io_tools.write_param_ln(
            result_dict,
            "system@gamma",
            "Centre of mass velocity (gamma):",
            write_fn,
            line_sep,
        )
        io_tools.write_param_ln(
            result_dict,
            "system@period",
            "Orbital period (P):",
            write_fn,
            line_sep,
        )
        if "system@primary_minimum_time" in result_dict:
            io_tools.write_param_ln(
                result_dict,
                "system@primary_minimum_time",
                "Time of primary minimum (T0):",
                write_fn,
                line_sep,
            )

        write_fn(f"{'-' * DASH_N}{line_sep}")

        if result_dict.get("r_squared", False):
            if result_dict["r_squared"]["value"] is not None:
                io_tools.write_param_ln(
                    result_dict,
                    "r_squared",
                    "Fit R^2: ",
                    write_fn,
                    line_sep,
                    6,
                )

            write_fn(f"{'-' * DASH_N}{line_sep}")
    finally:
        if f is not None:
            f.close()


def fit_rv_summary_with_error_propagation(
    fit_instance: RVFitMCMC,
    path: str | None,
    percentiles: list[int],
) -> None:
    """Propagate errors of RV fitted parameters from MCMC chain.

    Performs error propagation using provided MCMC results to generate
    comprehensive summary with confidence intervals for radial velocity
    fitting parameters.

    :param fit_instance: Radial velocity MCMC fit instance.
    :type fit_instance: RVFitMCMC
    :param path: Path to store summary. If None, output goes to stdout.
    :type path: str | None
    :param percentiles: List of percentiles [lower, center, upper] for
        confidence intervals, e.g., [16, 50, 84] for 1-sigma.
    :type percentiles: list[int]
    :return: None. Summary written to file or stdout.
    :rtype: None
    :raises ValueError: If MCMC chain is not found in fit_instance.
    """
    if path is not None:
        f = Path(path).open("w")  # noqa: SIM115
        write_fn: Callable[..., Any] = f.write
        line_sep: str = "\n"
    else:
        write_fn = print
        line_sep = ""

    flat_chain: NDArray | None = fit_instance.flat_chain
    variable_labels: list[str] = fit_instance.variable_labels
    normalization: dict[str, tuple[Float, Float]] = fit_instance.normalization

    flat_params: dict[str, Any] = fit_instance.flat_result
    params: dict[str, Any] = fit_instance.result

    params_instance: BinaryInitialParameters = BinaryInitialParameters(**params)
    fit_instance.fit_method_instance.fixed = params_instance.get_fixed(jsonify=False)
    fit_instance.fit_method_instance.constrained = params_instance.get_constrained(jsonify=False)
    fit_instance.fit_method_instance.fitable = params_instance.get_fitable(jsonify=False)

    error_msg: str = "MCMC chain was not found."
    if flat_chain is None:
        raise ValueError(error_msg)

    renormalized_chain: NDArray = np.empty(flat_chain.shape)
    for ii, lbl in enumerate(variable_labels):
        renormalized_chain[:, ii] = parameters.renormalize_value(
            flat_chain[:, ii],
            normalization[lbl][0],
            normalization[lbl][1],
        )
    stop_idx: dict[str, int] = {}

    complete_param_list: list[str] = [
        "system@mass_ratio",
        "system@asini",
        "system@eccentricity",
        "system@argument_of_periastron",
        "system@gamma",
        "system@period",
        "system@primary_minimum_time",
    ]

    param_columns: dict[str, int] = {lbl: ii for ii, lbl in enumerate(complete_param_list)}

    args: tuple[Any, ...] = (fit_instance, param_columns, stop_idx)
    full_chain: NDArray = _manage_chain_evaluation(renormalized_chain, evaluate_rv_params, *args)

    full_chain_mask: NDArray = (~np.isnan(full_chain)).any(axis=1)
    full_chain = full_chain[full_chain_mask]

    # avoiding np warnings about NANs
    calculated_percentiles: NDArray = np.empty((3, full_chain.shape[1]))
    calculated_percentiles[:] = up.NaN
    full_chain_mask = (~np.isnan(full_chain)).any(axis=0)
    calculated_percentiles[:, full_chain_mask] = np.percentile(full_chain[:, full_chain_mask], percentiles, axis=0)
    full_chain_results: NDArray = np.vstack(
        (
            calculated_percentiles[1, :],
            calculated_percentiles[1, :] - calculated_percentiles[0, :],
            calculated_percentiles[2, :] - calculated_percentiles[1, :],
        ),
    )

    intro: tuple[Callable[..., Any], str, str, str, str, str, str, str] = (
        write_fn,
        "Parameter",
        "value",
        "-1 sigma",
        "+1 sigma",
        "unit",
        "status",
        line_sep,
    )
    write_fn(f"\nBINARY SYSTEM{line_sep}")
    io_tools.write_ln(*intro)
    write_fn(f"{'-' * DASH_N}{line_sep}")

    io_tools.write_propagated_ln(
        full_chain_results[:, param_columns["system@mass_ratio"]],
        flat_params,
        "system@mass_ratio",
        "Mass ratio (q=M_2/M_1):",
        write_fn,
        line_sep,
        "-",
    )

    # sma = (full_chain_results[:, param_columns['system@asini']] *
    #        u.DISTANCE_UNIT).to(u.solRad).value
    sma = full_chain_results[:, param_columns["system@asini"]]
    io_tools.write_propagated_ln(
        sma,
        flat_params,
        "system@asini",
        "a*sin(i):",
        write_fn,
        line_sep,
        "solRad",
    )

    io_tools.write_propagated_ln(
        full_chain_results[:, param_columns["system@eccentricity"]],
        flat_params,
        "system@eccentricity",
        "Eccentricity (e):",
        write_fn,
        line_sep,
        "-",
    )

    omega = (
        (
            full_chain_results[:, param_columns["system@argument_of_periastron"]]
            * u.DefaultBinarySystemUnits.system.argument_of_periastron
        )
        .to(u.deg)
        .value
    )
    io_tools.write_propagated_ln(
        omega,
        flat_params,
        "system@argument_of_periastron",
        "Argument of periastron (omega):",
        write_fn,
        line_sep,
        "deg",
    )

    omega = (
        (full_chain_results[:, param_columns["system@gamma"]] * u.DefaultBinarySystemUnits.system.gamma)
        .to(u.km / u.s)
        .value
    )
    io_tools.write_propagated_ln(
        omega,
        flat_params,
        "system@gamma",
        "Centre of mass velocity (gamma):",
        write_fn,
        line_sep,
        "km/s",
    )

    io_tools.write_propagated_ln(
        full_chain_results[:, param_columns["system@period"]],
        flat_params,
        "system@period",
        "Orbital period (P):",
        write_fn,
        line_sep,
        "d",
    )

    if "system@primary_minimum_time" in flat_params:
        io_tools.write_param_ln(
            flat_params,
            "system@primary_minimum_time",
            "Time of primary minimum (T0):",
            write_fn,
            line_sep,
        )

    write_fn(f"{'-' * DASH_N}{line_sep}")


def evaluate_rv_params(*args: Any) -> NDArray:
    """Evaluate RV chain for error propagation in fit summary.

    Produces full set of model parameters from MCMC chain by constructing
    radial velocity system instances and extracting parameter values.

    :param args: Tuple containing (fit_instance, param_columns, stop_idx,
        renormalized_chain). See source for parameter descriptions.
    :type args: Any
    :return: Array of evaluated parameters with shape
        (n_chain_samples, n_parameters).
    :rtype: NDArray
    """
    fit_instance: Any
    param_columns: dict[str, int]
    renormalized_chain: NDArray

    fit_instance, param_columns, _, renormalized_chain = args
    full_chain: NDArray = np.empty((renormalized_chain.shape[0], len(param_columns)))
    for ii in tqdm(range(renormalized_chain.shape[0])):
        init_rv_kwargs: dict[str, Any] = parameters.prepare_properties_set(
            renormalized_chain[ii, :],
            fit_instance.fit_method_instance.fitable.keys(),
            fit_instance.fit_method_instance.constrained,
            fit_instance.fit_method_instance.fixed,
        )
        system_kwgs: dict[str, Any] = serializers.serialize_system_kwargs(**init_rv_kwargs)

        binary_instance: RadialVelocitySystem = RadialVelocitySystem(**RadialVelocitySystem.prepare_json(system_kwgs))
        for var_label in list(param_columns.keys())[:-1]:
            full_chain[ii, param_columns[var_label]] = getattr(binary_instance, var_label.split("@")[1], None)
        full_chain[ii, param_columns["system@primary_minimum_time"]] = init_rv_kwargs["system@primary_minimum_time"]

    return full_chain


def _manage_chain_evaluation(
    renormalized_chain: NDArray,
    eval_function: Callable,
    *args: Any,
) -> NDArray:
    """Manage evaluations of system parameters from MCMC chains.

    Enables use of multiprocessing for efficient evaluation of large MCMC
    chains. Splits chain into batches and evaluates them in parallel if
    configured, otherwise evaluates sequentially.

    :param renormalized_chain: Flat MCMC chain with normalized values.
    :type renormalized_chain: NDArray
    :param eval_function: Callable to evaluate chain batches. Signature:
        eval_function(fit_instance, ..., chain_batch) -> parameters_array.
    :type eval_function: Any
    :param args: Additional arguments passed to eval_function before chain.
    :type args: Any
    :return: Array of evaluated model parameters concatenated from all
        batches.
    :rtype: NDArray
    """
    if settings.NUMBER_OF_MCMC_PROCESSES > 1:
        logger.info("starting multiprocessor workers for error propagation technique")
        batches: list[NDArray] = split_to_batches(array=renormalized_chain, n_proc=settings.NUMBER_OF_MCMC_PROCESSES)
        pool: Pool = Pool(processes=settings.NUMBER_OF_MCMC_PROCESSES)
        result: list[Any] = [pool.apply_async(eval_function, (*args, batch)) for batch in batches]
        pool.close()
        pool.join()
        result = [r.get() for r in result]
        return np.vstack(result)
    return eval_function(*args, renormalized_chain)
