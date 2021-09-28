import numpy as np

from .. import utils as butils
from ... import atm, ld, const
from ... import settings
from ... observer.passband import init_bolometric_passband
from ... binary_system import radius as bsradius


def get_limbdarkening_cfs(system, component="all", **kwargs):
    """
    Returns limb darkening coefficients for each face of each component.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param component: str;
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float

    :return: Dict[str, numpy.array];
    """
    components = butils.component_to_list(component)

    symmetry_test = not system.has_spots() and not system.has_pulsations()
    temperatures, log_g = dict(), dict()

    for c_name in components:
        component_instance = getattr(system, c_name)
        if settings.USE_SINGLE_LD_COEFFICIENTS:
            temperatures[c_name] = np.array([component_instance.t_eff, ])
            log_g[c_name] = np.array([np.max(component_instance.log_g), ])
        elif symmetry_test:
            temperatures[c_name] = component_instance.symmetry_faces(component_instance.temperatures)
            log_g[c_name] = component_instance.symmetry_faces(component_instance.log_g)
        else:
            temperatures[c_name] = component_instance.temperatures
            log_g[c_name] = component_instance.log_g

    ld_cfs = {
        component:
            ld.interpolate_on_ld_grid(
                temperature=temperatures[component],
                log_g=log_g[component],
                metallicity=getattr(system, component).metallicity,
                passband=kwargs["passband"]
            ) for component in components
    }
    # mirroring symmetrical part back to the rest of the surface
    if symmetry_test:
        for c_name in components:
            if settings.USE_SINGLE_LD_COEFFICIENTS:
                ld_cfs[c_name] = {fltr: vals[np.zeros(getattr(system, c_name).temperatures.shape, dtype=np.int)]
                                  for fltr, vals in ld_cfs[c_name].items()}
            else:
                ld_cfs[c_name] = {fltr: getattr(system, c_name).mirror_face_values(vals)
                                  for fltr, vals in ld_cfs[c_name].items()}
    return ld_cfs


def _get_normal_radiance(system, component="all", **kwargs):
    """
    Compute normal radiance for all faces and all components
    in elisa.binary_system.container.OrbitalPositionContainer.

    :param component: str;
    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param kwargs: Dict; arguments to be passed into light curve generator functions
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float

    :return: Dict[String, dict]
    """
    components = butils.component_to_list(component)
    symmetry_test = {
        component: not getattr(system, component).has_spots() and not getattr(system, component).has_pulsations()
        for component in components
    }
    temperatures, log_g = dict(), dict()

    # utilizing surface symmetry in case of a clear surface
    for cmpnt in components:
        component_instance = getattr(system, cmpnt)
        if symmetry_test[cmpnt]:
            temperatures[cmpnt] = component_instance.symmetry_faces(component_instance.temperatures)
            log_g[cmpnt] = component_instance.symmetry_faces(component_instance.log_g)
        else:
            temperatures[cmpnt] = component_instance.temperatures
            log_g[cmpnt] = component_instance.log_g

    retval = {
        component:
            atm.NaiveInterpolatedAtm.radiance(
                **dict(
                    temperature=temperatures[component],
                    log_g=log_g[component],
                    metallicity=getattr(system, component).metallicity,
                    atlas=getattr(getattr(system, component), "atmosphere") or settings.ATM_ATLAS,
                    **kwargs
                )
            ) for component in components
    }

    # mirroring symmetrical part back to the rest of the surface
    for cpmnt in components:
        if symmetry_test[cpmnt]:
            retval[cpmnt] = {fltr: getattr(system, cpmnt).mirror_face_values(vals) for
                             fltr, vals in retval[cpmnt].items()}

    return retval


def prep_surface_params(system, return_values=True, write_to_containers=False, **kwargs):
    """
    Prepares normal radiances and limb darkening coefficients variables.

    :param system: elisa.binary_system.container.OrbitalPositionContainer;
    :param return_values: bool; return normal radiances and limb darkening coefficients
    :param write_to_containers: bool; calculated values will be assigned to `system` container
    :param kwargs: Dict;
    :**kwargs options**:
        * ** passband ** * - Dict[str, elisa.observer.PassbandContainer]
        * ** left_bandwidth ** * - float
        * ** right_bandwidth ** * - float
    :return: Tuple;
    """
    # obtain limb darkening factor for each face
    ld_cfs = get_limbdarkening_cfs(system, **kwargs)
    # compute normal radiance for each face and each component
    normal_radiance = _get_normal_radiance(system, **kwargs)

    # checking if `bolometric` filter is already used
    if 'bolometric' in ld_cfs['primary']:
        bol_ld_cfs = {
            component: {'bolometric': ld_cfs[component]['bolometric']}
            for component in settings.BINARY_COUNTERPARTS
        }
    else:
        passband, left_bandwidth, right_bandwidth = init_bolometric_passband()
        bol_kwargs = {
            'passband': {'bolometric': passband},
            'left_bandwidth': left_bandwidth,
            'right_bandwith': right_bandwidth,
            'atlas': 'whatever'
        }
        bol_ld_cfs = get_limbdarkening_cfs(system, **bol_kwargs)

    normal_radiance = atm.correct_normal_radiance_to_optical_depth(normal_radiance, bol_ld_cfs)

    if write_to_containers:
        for component in settings.BINARY_COUNTERPARTS:
            star = getattr(system, component)
            setattr(star, 'normal_radiance', normal_radiance[component])
            setattr(star, 'ld_cfs', ld_cfs[component])

    return normal_radiance, ld_cfs if return_values else None


def update_surface_params(require_rebuild, container, normal_radiance, ld_cfs, **kwargs):
    """
    Function either recalculates normal radiances and limb darkening coefficients or it assigns old values to the
    container according to the `require_rebuild` condition.

    :param require_rebuild: bool; testing condition for recalculation of surface parameters
    :param container: elisa.binary_system.container.OrbitalPositionContainer;
    :param normal_radiance: Dict; old values of normal radiances
    :param ld_cfs: Dict; old values of limb darkening coefficients
    :param kwargs: Dict;
    :return: Tuple; updated container and updated normal radiances and limb darkening coefficients
    """
    if require_rebuild:
        normal_radiance, ld_cfs = \
            prep_surface_params(container, return_values=True, write_to_containers=True, **kwargs)
    else:
        for component in settings.BINARY_COUNTERPARTS:
            star = getattr(container, component)
            setattr(star, 'normal_radiance', normal_radiance[component])
            setattr(star, 'ld_cfs', ld_cfs[component])

    return container, normal_radiance, ld_cfs


def split_orbit_by_apse_line(orbital_motion, orbital_mask):
    """
    Split orbital positions represented by `orbital_motion` array on two groups separated by line of apsides.
    Separation is defined by `orbital_mask`

    :param orbital_motion: numpy.array; arraywhcih represents orbital positions
    :param orbital_mask: numpy.array[bool]; mask which defines separation (True is one side and False is other side)
    :return: Tuple[numpy.array, numpy.array];
    """
    reduced_orbit_arr = orbital_motion[orbital_mask]
    supplement_to_reduced_arr = orbital_motion[~orbital_mask]
    return reduced_orbit_arr, supplement_to_reduced_arr


def forward_radii_from_distances(binary, distances, potentials=None):
    """
    Returns forward radii for each components distance.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param distances: numpy.array;
    :param potentials: numpy.array; if None, they will be calculated
    :return: numpy.array; 2*N array of forward radii for each component
    """
    corrected_potentials = binary.correct_potentials(distances=distances, component="all", iterations=2) \
        if potentials is None else potentials

    pargs = (distances, corrected_potentials['primary'], binary.mass_ratio, binary.primary.synchronicity, "primary")
    sargs = (distances, corrected_potentials['secondary'], binary.mass_ratio, binary.secondary.synchronicity,
             "secondary")

    return np.vstack((bsradius.calculate_forward_radii(*pargs), bsradius.calculate_forward_radii(*sargs)))


def compute_rel_d_geometry(binary, radii, radii_counterpart):
    """
    Function estimates the maximum change in flux due to the change in the geometry estimated by the change in the
    forward radius.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param radii: numpy.array;
    :param radii_counterpart: numpy.array;
    :return: numpy.array;
    """
    eq_radii = np.array([binary.primary.equivalent_radius, binary.secondary.equivalent_radius])
    fwd_r_diff = np.abs(radii_counterpart - radii)

    d_flux = 2 * eq_radii[:, np.newaxis] * fwd_r_diff + fwd_r_diff ** 2
    total_flux = eq_radii ** 2
    return d_flux / np.sum(total_flux)


def relative_irradiation(binary, distances):
    """
    Returns an estimate of how much of the component flux comes from the reflected light of the companion

    :param binary: elisa.binary_system.system.BinarySystem;
    :param distances: numpy.array; orbital distances
    :return: numpy.array;
    """
    temp_ratio4 = np.power(binary.primary.t_eff / binary.secondary.t_eff, 4)
    r_ratio2 = np.power(binary.primary.equivalent_radius / binary.secondary.equivalent_radius, 2)
    coeff = r_ratio2 * temp_ratio4
    irrad1 = np.power(binary.primary.equivalent_radius / distances, 2) / (1 + coeff)
    irrad2 = np.power(binary.secondary.equivalent_radius / distances, 2) / (1 + 1 / coeff)
    return np.vstack((irrad1, irrad2))


def compute_counterparts_rel_d_irrad(binary, distances, distances_counterpart):
    """
    Estimates a relative change in recieved irradiation from a companion.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param distances: numpy.array; orbital distances
    :param distances_counterpart: numpy.array; orbital distances
    :return: numpy.array;
    """
    irrad = relative_irradiation(binary, distances)
    irrad_c = relative_irradiation(binary, distances_counterpart)

    return np.abs(irrad - irrad_c)


def compute_rel_d_irradiation(binary, distances):
    """
    Estimates a relative change in recieved irradiation from a companion between the most similar orbital positions.

    :param binary: elisa.binary_system.system.BinarySystem;
    :param distances: numpy.array; orbital distances (sorted)
    :return: numpy.array;
    """
    irrad = relative_irradiation(binary, distances)
    return np.abs(irrad[:, 1:] - irrad[:, :-1])


def compute_rel_d_radii_from_counterparts(radii, base_positions, mirrors):
    """
    Returns relative differences between forward radii between two orbital counterparts.

    :param radii: numpy.array; forward radii
    :param base_positions: numpy.array; base orbital position array
    :param mirrors: numpy.array; orbital counterposition array
    :return: np.array; (2 * N) of relative changes in relative distances
    """
    fwd_radii_base = radii[base_positions[:, 0]]
    fwd_radii_counterpart = radii[mirrors[:, 0]]
    return np.abs(fwd_radii_base - fwd_radii_counterpart) / fwd_radii_base.mean(axis=1)[:, np.newaxis]


def prepare_apsidaly_symmetric_orbit(binary, azimuths, phases):
    """
    Prepare set of orbital positions that are symmetrical in therms of surface geometry. For each couple, the orbital
    position is mirrored using apsidal line in order to reduce time for generating the light curve.

    :param binary: elisa.binary_star.system.BinarySystem;
    :param azimuths: numpy.array; orbital azimuths of positions in which LC will be calculated
    :param phases: numpy.array; orbital phase of positions in which LC will be calculated
    :return: Tuple;


    shape ::

        (numpy.array, list, numpy.array)

        - unique_phase_indices - numpy.array : indices that points to the orbital positions from one half of the
        orbital motion divided by apsidal line
        - orbital_motion_counterpart - list - Positions produced by mirroring orbital positions given by
        indices `unique_phase_indices`
        - orbital_motion_array_counterpart - numpy.array - sa as `orbital_motion_counterpart` but in numpy.array form
    """
    azimuth_boundaries = [binary.argument_of_periastron, (binary.argument_of_periastron + const.PI) % const.FULL_ARC]

    if azimuth_boundaries[0] < azimuth_boundaries[1]:
        unique_geometry = np.logical_and(azimuths >= azimuth_boundaries[0], azimuths < azimuth_boundaries[1])
    else:
        unique_geometry = np.logical_xor(azimuths <= azimuth_boundaries[0], azimuths > azimuth_boundaries[1])

    unique_phase_indices = np.arange(phases.shape[0])[unique_geometry]
    unique_geometry_azimuths = azimuths[unique_geometry]
    unique_geometry_counterazimuths = (2 * binary.argument_of_periastron - unique_geometry_azimuths) % const.FULL_ARC

    kwargs = dict(input_argument=unique_geometry_counterazimuths, return_nparray=True, calculate_from='azimuth')
    orbital_motion_array_counterpart = binary.calculate_orbital_motion(**kwargs)

    return unique_phase_indices, orbital_motion_array_counterpart, unique_geometry


def adjust_eclipse_width(true_anomalies, true_anomaly_of_eclipse):
    """
    Extends the angular width of the eclipse by the separation of the true anomalies near the eclipse to smooth out 
    the transition before/after the eclipse.
    
    :param true_anomalies: numpy.array; true anomalies of the orbital positions
    :param true_anomaly_of_eclipse: float; true anomaly of the eclipse
    :return: float; adjusted width of an eclipse
    """
    distances = np.abs(true_anomalies - true_anomaly_of_eclipse)
    inverse_points_mask = distances > const.PI
    distances[inverse_points_mask] = const.FULL_ARC - distances[inverse_points_mask]
    idxs = np.argsort(distances)[:2]
    return 1.5 * np.abs(true_anomalies[idxs[1]] - true_anomalies[idxs[0]])
