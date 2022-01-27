import numpy as np
from ... import settings, ld, umpy as up
from ...observer.passband import init_rv_passband


def include_passband_data_to_kwargs(**kwargs):
    """
    Including dummy passband from which radiometric radial velocities will be calculated.

    :param kwargs: Tuple;
    :return: Tuple;
    """
    psbnd, right_bandwidth, left_bandwidth = init_rv_passband()
    kwargs.update({
        'passband': {'rv_band': psbnd},
        'left_bandwidth': left_bandwidth,
        'right_bandwidth': right_bandwidth
    })
    return kwargs


def calculate_surface_element_fluxes(band, star):
    """
    Function generates outgoing flux from each surface element of given star container in certain band.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :param band: str; name of the photometric band compatible with supported names in config
    :return: numpy.array
    """
    indices = star.indices
    radiance = star.normal_radiance[band][indices]
    ld_cfs = star.ld_cfs[band][indices]
    cosines = star.los_cosines[indices]
    coverage = star.coverage[indices]

    ld_cors = ld.limb_darkening_factor(coefficients=ld_cfs,
                                       limb_darkening_law=settings.LIMB_DARKENING_LAW,
                                       cos_theta=cosines)

    return radiance * cosines * coverage * ld_cors


def flux_from_star_container(band, star):
    """
    Function generates outgoing flux from given star container in certain band.

    :param star: elisa.base.container.StarContainer; star container with all necessary parameters pre-calculated
    :param band: str; name of the photometric band compatible with supported names in config,
    :return: numpy.array;
    """

    return up.sum(calculate_surface_element_fluxes(band, star))


def generate_teff_logg_for_ld_cfs(component_instance, symmetry_test):
    """
    Generates temperatures and log_g parameters either for full star or symmetrical part based on the symmetry test.

    :param component_instance: elisa.base.container.StarContainer; star container
    :param symmetry_test: bool; whether to use star symmetry
    :return: Tuple;
    """
    if settings.USE_SINGLE_LD_COEFFICIENTS:
        temperatures = np.array([component_instance.t_eff, ])
        log_g = np.array([np.max(component_instance.log_g), ])
    elif symmetry_test:
        temperatures = component_instance.symmetry_faces(component_instance.temperatures)
        log_g = component_instance.symmetry_faces(component_instance.log_g)
    else:
        temperatures = component_instance.temperatures
        log_g = component_instance.log_g

    return temperatures, log_g


def get_component_limbdarkening_cfs(component_instance, symmetry_test, passbands):
    if component_instance.limb_darkening_coefficients is not None:
        desired_repeats = (component_instance.temperatures.shape[0], 1)
        ld_cfs = {passband: np.tile(component_instance.limb_darkening_coefficients, desired_repeats)
                  for passband in passbands}
    else:
        temperatures, log_g = generate_teff_logg_for_ld_cfs(component_instance, symmetry_test)

        ld_cfs = ld.interpolate_on_ld_grid(
            temperature=temperatures,
            log_g=log_g,
            metallicity=component_instance.metallicity,
            passband=passbands
        )

        if symmetry_test:
            if settings.USE_SINGLE_LD_COEFFICIENTS:
                ld_cfs = {fltr: vals[np.zeros(component_instance.temperatures.shape, dtype=np.int)]
                          for fltr, vals in ld_cfs.items()}
            else:
                ld_cfs = {fltr: component_instance.mirror_face_values(vals)
                          for fltr, vals in ld_cfs.items()}

    return ld_cfs
