from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

from elisa import BinarySystem, Observer, Star

PASSBAND = "Generic.Bessell.U"


@dataclass
class GenerationConfig:
    """Configuration for synthetic light-curve generation."""

    from_phase: float = -0.6
    to_phase: float = 0.6
    phase_step: float = 0.002

    regular_grid: bool = False
    n_random_points: int = 400
    random_seed: int = 42

    relative_error: float = 5e-4
    absolute_error_floor: float = 2e-3
    use_flux_dependent_errors: bool = True

    time_offset: float = 2400000.0
    output_file: str = "../../../synthetic_bessell_U.dat"
    normalize_flux: bool = True


@dataclass
class ModelData:
    """Container for the smooth model light curve."""

    phases: np.ndarray
    fluxes: np.ndarray
    times_out: np.ndarray


@dataclass
class SampledData:
    """Container for sampled synthetic observations."""

    phases: np.ndarray
    flux_model: np.ndarray
    flux_noisy: np.ndarray
    flux_err: np.ndarray
    times_out: np.ndarray


def build_binary_system() -> BinarySystem:
    """Create the binary-system model used for synthetic data generation."""
    primary = Star(
        mass=2.15 * u.solMass,
        surface_potential=3.6,
        synchronicity=1.0,
        t_eff=10000 * u.K,
        gravity_darkening=1.0,
        discretization_factor=5,  # Angular size (deg) of surface elements.
        albedo=0.6,
        metallicity=0.0,
        atmosphere="bb",
    )

    secondary = Star(
        mass=1.2 * u.solMass,
        surface_potential=4.0,
        synchronicity=1.0,
        t_eff=7000 * u.K,
        gravity_darkening=1.0,
        albedo=0.6,
        metallicity=0.0,
    )

    primary_minimum_time_jd = 2454953.5388437

    return BinarySystem(
        primary=primary,
        secondary=secondary,
        argument_of_periastron=58 * u.deg,
        gamma=-30.7 * u.km / u.s,
        period=2.5 * u.d,
        eccentricity=0.0,
        inclination=85 * u.deg,
        distance=155 * u.pc,
        primary_minimum_time=primary_minimum_time_jd * u.d,
        phase_shift=0.0,
    )


def build_observer(system: BinarySystem) -> Observer:
    """Create an observer for the selected passband."""
    return Observer(
        passband=[PASSBAND],
        system=system,
    )


def generate_model_curve(
    observer: Observer,
    config: GenerationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a smooth model light curve on a phase grid."""
    phases_model, fluxes_model = observer.observe.lc(
        from_phase=config.from_phase,
        to_phase=config.to_phase,
        phase_step=config.phase_step,
        normalize=config.normalize_flux,
    )

    phases_model = np.asarray(phases_model, dtype=float)

    if isinstance(fluxes_model, dict):
        flux_model_array = np.asarray(fluxes_model[PASSBAND], dtype=float)
    else:
        flux_model_array = np.asarray(fluxes_model, dtype=float)

    return phases_model, flux_model_array


def sample_model_curve(
    phases_model: np.ndarray,
    flux_model: np.ndarray,
    config: GenerationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the smooth model either on the original grid or at random phases."""
    rng = np.random.default_rng(config.random_seed)

    if config.regular_grid:
        phases_sample = phases_model.copy()
        flux_sample_model = flux_model.copy()
    else:
        phases_sample = np.sort(
            rng.uniform(config.from_phase, config.to_phase, config.n_random_points),
        )
        flux_sample_model = np.interp(phases_sample, phases_model, flux_model)

    return phases_sample, flux_sample_model


def phases_to_times(
    phases: np.ndarray,
    period_days: float,
    t0_jd: float,
    time_offset: float,
) -> np.ndarray:
    """Convert orbital phases to output times."""
    times_jd = t0_jd + phases * period_days
    return times_jd - time_offset


def estimate_flux_errors(
    flux_model: np.ndarray,
    config: GenerationConfig,
) -> np.ndarray:
    """Estimate per-point flux uncertainties."""
    if config.use_flux_dependent_errors:
        return np.maximum(
            config.absolute_error_floor,
            config.relative_error * flux_model,
        )

    return np.full_like(flux_model, config.absolute_error_floor, dtype=float)


def add_noise(
    flux_model: np.ndarray,
    flux_err: np.ndarray,
    random_seed: int,
) -> np.ndarray:
    """Add Gaussian noise to the model flux."""
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(loc=0.0, scale=flux_err, size=flux_err.shape)
    return flux_model + noise


def sort_by_time(
    times_out: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Sort one or more arrays by the provided time axis."""
    idx = np.argsort(times_out)
    sorted_arrays = [times_out[idx]]
    sorted_arrays.extend(array[idx] for array in arrays)
    return tuple(sorted_arrays)


def write_dat_file(
    output_file: str,
    times_out: np.ndarray,
    fluxes: np.ndarray,
    flux_err: np.ndarray,
) -> None:
    """Write synthetic light-curve data to a .dat file."""
    with Path(output_file).open("w", encoding="utf-8") as file:
        file.write("#time  flux    error\n")
        file.writelines(
            f"{time_value:.7f}    {flux_value:.12f}    {err_value:.6g}\n"
            for time_value, flux_value, err_value in zip(times_out, fluxes, flux_err, strict=True)
        )


def print_summary(
    config: GenerationConfig,
    n_points: int,
    period_days: float,
    t0_jd: float,
) -> None:
    """Print a brief summary of the generated dataset."""
    t0_out = t0_jd - config.time_offset

    print(f"Done. Output file: {config.output_file}")  # noqa: T201
    print(f"Number of points: {n_points}")  # noqa: T201
    print(f"Period (d): {period_days}")  # noqa: T201
    print(f"T0 (JD): {t0_jd}")  # noqa: T201
    print(f"T0 (output #time): {t0_out}")  # noqa: T201
    print(f"Time offset: {config.time_offset}")  # noqa: T201


def plot_synthetic_data(
    model_data: ModelData,
    sampled_data: SampledData,
    config: GenerationConfig,
    t0_out: float,
) -> None:
    """Plot the smooth model together with noisy sampled data and error bars."""
    plt.figure(figsize=(11, 6))

    plt.plot(
        model_data.times_out,
        model_data.fluxes,
        linewidth=1.5,
        label="model",
    )

    plt.errorbar(
        sampled_data.times_out,
        sampled_data.flux_noisy,
        yerr=sampled_data.flux_err,
        fmt="none",
        elinewidth=0.8,
        capsize=0,
        alpha=0.8,
        label="flux error",
    )

    plt.scatter(
        sampled_data.times_out,
        sampled_data.flux_noisy,
        s=12,
        alpha=0.9,
        label="synthetic data",
    )

    plt.axvline(
        t0_out,
        linestyle="--",
        linewidth=1.0,
        label="T0",
    )

    plt.xlabel(f"time [JD - {config.time_offset:g}]")
    plt.ylabel("normalized flux")
    plt.title(f"Synthetic light curve in {PASSBAND}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Generate synthetic light-curve data and save them to disk."""
    config = GenerationConfig()

    binary_system = build_binary_system()
    observer = build_observer(binary_system)

    phases_model, flux_model = generate_model_curve(observer, config)
    phases_sample, flux_sample_model = sample_model_curve(
        phases_model=phases_model,
        flux_model=flux_model,
        config=config,
    )

    period_days = float(binary_system.period)
    t0_jd = float(binary_system.primary_minimum_time)
    t0_out = t0_jd - config.time_offset

    times_out_model = phases_to_times(
        phases=phases_model,
        period_days=period_days,
        t0_jd=t0_jd,
        time_offset=config.time_offset,
    )
    times_out_sample = phases_to_times(
        phases=phases_sample,
        period_days=period_days,
        t0_jd=t0_jd,
        time_offset=config.time_offset,
    )

    flux_err = estimate_flux_errors(flux_sample_model, config)
    flux_noisy = add_noise(
        flux_model=flux_sample_model,
        flux_err=flux_err,
        random_seed=config.random_seed,
    )

    times_out_model, phases_model, flux_model = sort_by_time(
        times_out_model,
        phases_model,
        flux_model,
    )
    (
        times_out_sample,
        phases_sample,
        flux_sample_model,
        flux_noisy,
        flux_err,
    ) = sort_by_time(
        times_out_sample,
        phases_sample,
        flux_sample_model,
        flux_noisy,
        flux_err,
    )

    model_data = ModelData(
        phases=phases_model,
        fluxes=flux_model,
        times_out=times_out_model,
    )
    sampled_data = SampledData(
        phases=phases_sample,
        flux_model=flux_sample_model,
        flux_noisy=flux_noisy,
        flux_err=flux_err,
        times_out=times_out_sample,
    )

    # write_dat_file(
    #     output_file=config.output_file,
    #     times_out=sampled_data.times_out,
    #     fluxes=sampled_data.flux_noisy,
    #     flux_err=sampled_data.flux_err,
    # )

    print_summary(
        config=config,
        n_points=len(sampled_data.times_out),
        period_days=period_days,
        t0_jd=t0_jd,
    )

    plot_synthetic_data(
        model_data=model_data,
        sampled_data=sampled_data,
        config=config,
        t0_out=t0_out,
    )


if __name__ == "__main__":
    main()
