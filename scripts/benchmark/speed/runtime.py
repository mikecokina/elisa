"""Benchmark light curve generation performance with varying discretization and phases.

This script measures execution time for light curve generation across different
surface discretization factors and orbital phase counts for circular and eccentric
binary systems.
"""  # noqa: INP001

from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

import numpy as np

from elisa.binary_system import system
from elisa.conf import settings
from elisa.observer.observer import Observer

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_params(filename: Path | str) -> dict[str, Any]:
    """Load binary system parameters from JSON file.

    :param filename: Path to JSON file containing binary system configuration.
    :type filename: Path | str
    :returns: Dictionary with binary system parameters.
    :rtype: dict[str, Any]
    """
    filepath = Path(filename)
    with filepath.open() as f:
        return json.loads(f.read())


def get_data(
    data: dict[str, Any],
    phs: NDArray[np.floating],
) -> None:
    """Compute light curve for given phases.

    :param data: Binary system configuration dictionary.
    :type data: dict[str, Any]
    :param phs: Array of orbital phases to compute light curve at.
    :type phs: NDArray[np.floating]
    :returns: None
    :rtype: None
    """
    binary = system.BinarySystem.from_json(data)
    observer = Observer(
        passband=[
            "Generic.Bessell.B",
            "Generic.Bessell.V",
            "Generic.Bessell.R",
        ],
        system=binary,
    )
    _ = observer.lc(phases=phs)


def print_header(
    table_width: int) -> None:
    """Print table header with column names.

    :param table_width: Total width of the table in characters.
    :type table_width: int
    :returns: None
    :rtype: None
    """
    print("\n" + "=" * table_width)  # noqa: T201
    print(  # noqa: T201
        f"{'Orbit Type':<12} "
        f"{'Disc':<6} "
        f"{'Phases':<8} "
        f"{'Mean (s)':<12} "
        f"{'Std (s)':<12} "
        f"{'Min (s)':<12} "
        f"{'Max (s)':<12}",
    )
    print("-" * table_width)  # noqa: T201


def print_row(
    orbit_type: str,
    alpha: int,
    phs: int,
    times: list[float],
) -> None:
    """Print a formatted table row with statistics.

    :param orbit_type: Type of orbit (circular/eccentric).
    :type orbit_type: str
    :param alpha: Discretization factor.
    :type alpha: int
    :param phs: Number of phases.
    :type phs: int
    :param times: List of individual run times in seconds.
    :type times: list[float]
    :returns: None
    :rtype: None
    """
    mean_time = float(np.mean(times))
    std_time = float(np.std(times))
    min_time = float(np.min(times))
    max_time = float(np.max(times))

    print(  # noqa: T201
        f"{orbit_type:<12} "
        f"{alpha:<6} "
        f"{phs:<8} "
        f"{mean_time:<12.4f} "
        f"{std_time:<12.4f} "
        f"{min_time:<12.4f} "
        f"{max_time:<12.4f}",
    )


def _benchmark_eccentric(
    data_circ: dict[str, Any],
    data_ecc: dict[str, Any],
    outfl1: Path,
    outfl2: Path,
    surface_discretizations: list[int],
    n_phases: list[int],
    num_repetitions: int,
    table_width: int,
) -> None:
    """Run benchmark for both circular and eccentric orbits.

    :param data_circ: Circular orbit data dictionary.
    :type data_circ: dict[str, Any]
    :param data_ecc: Eccentric orbit data dictionary.
    :type data_ecc: dict[str, Any]
    :param outfl1: Output file path for circular results.
    :type outfl1: Path
    :param outfl2: Output file path for eccentric results.
    :type outfl2: Path
    :param surface_discretizations: List of discretization factors.
    :type surface_discretizations: list[int]
    :param n_phases: List of phase counts.
    :type n_phases: list[int]
    :param num_repetitions: Number of runs per measurement.
    :type num_repetitions: int
    :param table_width: Table width for output.
    :type table_width: int
    :returns: None
    :rtype: None
    """
    with outfl1.open("w") as f:
        f.write("#Alpha    n_phases     mean_time  std_time   min_time   max_time\n")
        with outfl2.open("w") as g:
            g.write("#Alpha    n_phases     mean_time  std_time   min_time   max_time\n")

            for alpha in surface_discretizations:
                data_circ["primary"]["discretization_factor"] = alpha
                data_ecc["primary"]["discretization_factor"] = alpha
                print_header(table_width)

                for phs in n_phases:
                    # Benchmark circular orbit
                    circ_times: list[float] = []
                    for _ in range(num_repetitions):
                        start_time = time()
                        phases = np.linspace(-0.5, 0.5, num=phs)
                        get_data(data_circ, phases)
                        circ_times.append(time() - start_time)

                    circ_mean = float(np.mean(circ_times))
                    circ_std = float(np.std(circ_times))
                    circ_min = float(np.min(circ_times))
                    circ_max = float(np.max(circ_times))
                    f.write(
                        f"{alpha:>5} {phs:>10} {circ_mean:>10.4f} "
                        f"{circ_std:>10.4f} {circ_min:>10.4f} {circ_max:>10.4f}\n",
                    )
                    print_row("circular", alpha, phs, circ_times)

                    # Benchmark eccentric orbit
                    ecc_times: list[float] = []
                    for _ in range(num_repetitions):
                        start_time = time()
                        phases = np.linspace(-0.5, 0.5, num=phs)
                        get_data(data_ecc, phases)
                        ecc_times.append(time() - start_time)

                    ecc_mean = float(np.mean(ecc_times))
                    ecc_std = float(np.std(ecc_times))
                    ecc_min = float(np.min(ecc_times))
                    ecc_max = float(np.max(ecc_times))
                    g.write(
                        f"{alpha:>5} {phs:>10} {ecc_mean:>10.4f} "
                        f"{ecc_std:>10.4f} {ecc_min:>10.4f} {ecc_max:>10.4f}\n",
                    )
                    print_row("eccentric", alpha, phs, ecc_times)


def _benchmark_circular(
    data_circ: dict[str, Any],
    outfl1: Path,
    surface_discretizations: list[int],
    n_phases: list[int],
    num_repetitions: int,
    table_width: int,
) -> None:
    """Run benchmark for circular orbits only.

    :param data_circ: Circular orbit data dictionary.
    :type data_circ: dict[str, Any]
    :param outfl1: Output file path for circular results.
    :type outfl1: Path
    :param surface_discretizations: List of discretization factors.
    :type surface_discretizations: list[int]
    :param n_phases: List of phase counts.
    :type n_phases: list[int]
    :param num_repetitions: Number of runs per measurement.
    :type num_repetitions: int
    :param table_width: Table width for output.
    :type table_width: int
    :returns: None
    :rtype: None
    """
    with outfl1.open("w") as f:
        f.write("#Alpha    n_phases     mean_time  std_time   min_time   max_time\n")

        for alpha in surface_discretizations:
            data_circ["primary"]["discretization_factor"] = alpha
            print_header(table_width)

            for phs in n_phases:
                circ_times: list[float] = []
                for _ in range(num_repetitions):
                    start_time = time()
                    phases = np.linspace(-0.5, 0.5, num=phs)
                    get_data(data_circ, phases)
                    circ_times.append(time() - start_time)

                circ_mean = float(np.mean(circ_times))
                circ_std = float(np.std(circ_times))
                circ_min = float(np.min(circ_times))
                circ_max = float(np.max(circ_times))
                f.write(
                    f"{alpha:>5} {phs:>10} {circ_mean:>10.4f} "
                    f"{circ_std:>10.4f} {circ_min:>10.4f} {circ_max:>10.4f}\n",
                )
                print_row("circular", alpha, phs, circ_times)


def main() -> None:
    """Run the light curve generation benchmark.

    Measures LC computation time across varying discretization factors and
    orbital phase counts for circular and eccentric binary systems. Results
    are saved to files and printed to console in formatted tables.

    Configuration variables (edit these to customize the benchmark):
        num_repetitions: Number of individual runs per measurement
        compute_eccentric: Whether to benchmark eccentric orbits
        surface_discretizations: List of discretization factors to test
        n_phases: List of phase counts to test

    :returns: None
    :rtype: None
    """
    # =========================================================================
    # CONFIGURATION - Edit these variables to customize the benchmark
    # =========================================================================
    num_repetitions = 10  # Number of runs per measurement (increase for more reliable stats)
    compute_eccentric = True  # Set to False for benchmarking only circular orbits
    surface_discretizations = [10, 7, 5, 3]  # Discretization factors to test
    # n_phases = [50, 100, 150, 200, 250, 300, 350, 400]  # Phase counts to test
    n_phases = [50, 100]
    # =========================================================================

    # Configure settings
    settings.settings.configure(LOG_CONFIG="fit")
    settings.LIMB_DARKENING_LAW = "logarithmic"

    # Load binary system configurations
    data_circ = get_params("data/test_binary_circ.json")
    data_ecc = get_params("data/test_binary_ecc.json")
    outfl1 = Path("benchmark_circ.dat")
    outfl2 = Path("benchmark_ecc.dat")

    # Table formatting configuration
    table_width = 80

    print("\n" + "=" * table_width)  # noqa: T201
    print("LIGHT CURVE BENCHMARK - DISCRETIZATION AND PHASE COUNT ANALYSIS")  # noqa: T201
    print(  # noqa: T201
        f"Repetitions per measurement: {num_repetitions} | "
        f"Orbit types: {'Circular + Eccentric' if compute_eccentric else 'Circular Only'}",
    )
    print("=" * table_width)  # noqa: T201

    if compute_eccentric:
        _benchmark_eccentric(
            data_circ,
            data_ecc,
            outfl1,
            outfl2,
            surface_discretizations,
            n_phases,
            num_repetitions,
            table_width,
        )
    else:
        _benchmark_circular(
            data_circ,
            outfl1,
            surface_discretizations,
            n_phases,
            num_repetitions,
            table_width,
        )

    print("\n" + "=" * table_width)  # noqa: T201
    print(  # noqa: T201
        f"Results saved to: {outfl1}"
        + (f" and {outfl2}" if compute_eccentric else ""),
    )
    print("=" * table_width + "\n")  # noqa: T201


if __name__ == "__main__":
    main()

