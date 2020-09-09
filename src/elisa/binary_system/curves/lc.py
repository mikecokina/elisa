import numpy as np

from copy import (
    deepcopy,
)
from scipy.interpolate import Akima1DInterpolator

from ...logger import getLogger
from ...binary_system.container import OrbitalPositionContainer
from ...binary_system.orbit.container import OrbitalSupplements
from ...binary_system.surface.coverage import calculate_coverage_with_cosines
from ...binary_system.curves import (
    lcmp,
    curves,
    utils as crv_utils,
)

from ... import (
    umpy as up,
    const,
    utils
)
from ...binary_system import (
    utils as bsutils,
    dynamic,
)


logger = getLogger('binary_system.curves.lc')


def _onpos_params(on_pos, **kwargs):
    """
    Helper function.

    :param on_pos: elisa.binary_system.container.OrbitalPositionContainer;
    :return: Tuple;
    """
    _normal_radiance, _ld_cfs = crv_utils.prep_surface_params(on_pos, **kwargs)

    _coverage, _cosines = calculate_coverage_with_cosines(on_pos, on_pos.semi_major_axis, in_eclipse=True)
    return _normal_radiance, _ld_cfs, _coverage, _cosines


def _update_surface_in_ecc_orbits(system, orbital_position, new_geometry_test):
    """
    Function decides how to update surface properties with respect to the degree of change
    in surface geometry given by new_geometry test.
    If true, only points and normals are recalculated, otherwise surface is calculated from scratch.

    :param system: elisa.binary_system.container.OrbitalPositionContainer
    :param orbital_position:  OrbitalPosition list
    :param new_geometry_test: bool; test that will decide, how the following phase will be calculated
    :return: elisa.binary_system.system.BinarySystem; instance with updated geometry
    """
    if new_geometry_test:
        system.build(components_distance=orbital_position.distance)
    else:
        system.build_mesh(component="all", components_distance=orbital_position.distance)
        system.build_surface_areas(component="all")
        system.build_faces_orientation(component="all", components_distance=orbital_position.distance)

    return system


def compute_circular_synchronous_lightcurve(binary, **kwargs):
    """
    Compute light curve for synchronous circular binary system.


    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
            * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
            * ** left_bandwidth ** * - float
            * ** right_bandwidth ** * - float
            * ** atlas ** * - str
            * ** position_method** * - function definition; to evaluate orbital positions
            * ** phases ** * - numpy.array
    :return: Dict[str, numpy.array];
    """

    initial_system = curves.prep_initial_system(binary)

    phases = kwargs.pop("phases")
    unique_phase_interval, reverse_phase_map = dynamic.phase_crv_symmetry(initial_system, phases)

    lc_labels = list(kwargs["passband"].keys())

    band_curves = curves.produce_circ_sync_curves(binary, initial_system, unique_phase_interval,
                                                  lcmp.compute_lc_on_pos, lc_labels, **kwargs)

    band_curves = {band: band_curves[band][reverse_phase_map] for band in band_curves}
    return band_curves


def compute_circular_spotty_asynchronous_lightcurve(binary, **kwargs):
    """
    Function returns light curve of asynchronous systems with circular orbits and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
    :return: Dict; fluxes for each filter
    """
    lc_labels = list(kwargs["passband"].keys())

    return curves.produce_circ_spotty_async_curves(binary, lcmp.compute_lc_on_pos,
                                                   lc_labels, **kwargs)


def compute_eccentric_lightcurve_no_spots(binary, **kwargs):
    """
    General function for generating light curves of binaries with eccentric orbit and no spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs:  Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
    :return: Dict; fluxes for each filter
    """
    lc_labels = list(kwargs["passband"].keys())

    return curves.produce_ecc_curves_no_spots(binary, lcmp.compute_lc_on_pos, lc_labels, **kwargs)


def compute_eccentric_spotty_lightcurve(binary, **kwargs):
    """
    Function returns light curve of systems with eccentric orbits and spots.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** - float
        * ** right_bandwidth ** - float
        * ** atlas ** - str
    :return: Dict; dictionary of fluxes for each filter
    """
    lc_labels = list(kwargs["passband"].keys())

    return curves.produce_ecc_curves_with_spots(binary, lcmp.compute_lc_on_pos, lc_labels, **kwargs)
