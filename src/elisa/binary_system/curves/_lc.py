
def compute_eccentric_spotty_asynchronous_lightcurve(self, **kwargs):
    """
    Function returns light curve of assynchronous systems with eccentric orbits and spots.
    fixme: add params types

    :param self:
    :param kwargs:
    :return: dictionary of fluxes for each filter
    """

    phases = kwargs.pop("phases")
    position_method = kwargs.pop("position_method")
    orbital_motion = position_method(input_argument=phases, return_nparray=False, calculate_from='phase')

    # pre-calculate the longitudes of each spot for each phase
    spots_longitudes = geo.calculate_spot_longitudes(self, phases, component="all")

    # calculating lc with spots gradually shifting their positions in each phase
    band_curves = {key: np.empty(phases.shape) for key in kwargs["passband"]}

    # surface potentials with constant volume of components
    potentials = self.correct_potentials(phases, component="all", iterations=2)

    for ii, orbital_position in enumerate(orbital_motion):
        self.primary.surface_potential = potentials['primary'][ii]
        self.secondary.surface_potential = potentials['secondary'][ii]

        # assigning new longitudes for each spot
        geo.assign_spot_longitudes(self, spots_longitudes, index=ii, component="all")

        self.build(components_distance=orbital_position.distance)

        container = get_onpos_container(self, orbital_position, ecl_boundaries=None)

        normal_radiance = get_normal_radiance(container, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container, **kwargs)

        container.coverage, container.cosines = calculate_surface_parameters(container, self.semi_major_axis,
                                                                             in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = \
                calculate_lc_point(container, band, ld_cfs, normal_radiance)

    return band_curves
