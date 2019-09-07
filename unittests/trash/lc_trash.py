def __integrate_lc_appx_two(self, orbital_motion, missing_phases_indices, index_of_closest,
                          index_of_closest_reversed, uniq_geom_mask, ecl_boundaries, phases, new_geometry_test,
                          **kwargs):
    """
    function calculates LC for eccentric orbit for selected filters using approximation where to each OrbitalPosition on
    one side of the apsidal line, the closest counterpart OrbitalPosition is assigned and the same surface geometry is
    assumed for both of them.

    :param new_geometry_test: bool array - mask to indicate, during which orbital position, surface geometry should be
                                           recalculated
    :param self: BinarySystem instance
    :param orbital_motion: list of all OrbitalPositions at which LC will be calculated
    :param missing_phases_indices: if the number of phase curve is odd, or due to specific alligning of the phases along
    the orbit, the projection between two groups of the points is not necessarilly bijective. In such case
    `missing_phases_indices` point to the OrbitalPositions from approximated side of the orbit that doesnt have the
    counterpart on the other side of the apsidal line yet. This issue is remedied inside the function
    :param index_of_closest: list of indices that points to the counterpart OrbitalPositions on the approximated side of
    the orbit, The n-th index points to the conterpart of the n-th Orbital position on the exactly evaluated side of the
    orbit
    :param index_of_closest_reversed: for OrbitalPositions without counterpart, the index of the closest counterpart
    from the exactly evaluated side of the orbit is supplied
    :param uniq_geom_mask: boll array that is used as a mask to select orbital positions from one side of the apsidal
    line which LC points will be calculated exactly
    :param ecl_boundaries: list of phase boundaries of eclipses
    :param phases: phases in which the phase curve will be calculated
    :param kwargs: kwargs taken from `compute_eccentric_lightcurve` function
    :return: dictionary of fluxes for each filter
    """
    band_curves = {key: np.zeros(phases.shape) for key in kwargs["passband"]}

    template_phases_idx = np.arange(phases.shape[0])[uniq_geom_mask]
    orb_motion_template = [orbital_motion[ii] for ii in template_phases_idx]
    counterpart_phases_idx = np.arange(phases.shape[0])[~uniq_geom_mask]
    orb_motion_counterpart = [orbital_motion[ii] for ii in counterpart_phases_idx]

    phases_to_correct = phases[uniq_geom_mask]
    # appending orbital motion arrays to include missing phases to complete LC
    if len(missing_phases_indices) > 0:
        for ii, idx_reversed in enumerate(index_of_closest_reversed):
            orb_motion_template.append(orb_motion_template[idx_reversed])
            orb_motion_counterpart.append(orb_motion_counterpart[missing_phases_indices[ii]])

            phases_to_correct = np.append(phases_to_correct, phases_to_correct[idx_reversed])

    # surface potentials with constant volume of components
    potentials = self.correct_potentials(phases_to_correct, component=None, iterations=2)

    for counterpart_idx, orbital_position in enumerate(orb_motion_template):
        self.primary.surface_potential = potentials['primary'][counterpart_idx]
        self.secondary.surface_potential = potentials['secondary'][counterpart_idx]

        self = update_surface_in_ecc_orbits(self, orbital_position, new_geometry_test[counterpart_idx])

        orbital_position_counterpart = orb_motion_counterpart[index_of_closest[counterpart_idx]]

        container = get_onpos_container(self, orbital_position, ecl_boundaries)
        container_counterpart = get_onpos_container(self, orbital_position_counterpart, ecl_boundaries)

        normal_radiance = get_normal_radiance(container, **kwargs)
        ld_cfs = get_limbdarkening_cfs(container, **kwargs)

        container.coverage, container.cosines = calculate_surface_parameters(container, in_eclipse=True)
        container_counterpart.coverage, container_counterpart.cosines = \
            calculate_surface_parameters(container_counterpart, in_eclipse=True)

        for band in kwargs["passband"]:
            band_curves[band][int(orbital_position.idx)] = \
                calculate_lc_point(container, band, ld_cfs, normal_radiance)
            band_curves[band][int(orb_motion_counterpart[index_of_closest[counterpart_idx]].idx)] = \
                calculate_lc_point(container_counterpart, band, ld_cfs, normal_radiance)

    return band_curves








def __compute_eccentric_lightcurve(self, **kwargs):
    ecl_boundaries = geo.get_eclipse_boundaries(self, 1.0)

    phases = kwargs.pop("phases")
    position_method = kwargs.pop("position_method")

    om_params = dict(input_argument=phases, return_nparray=True, calculate_from='phase')
    all_orbital_pos_arr = position_method(**om_params)

    phases_span_test = np.max(phases) - np.min(phases) >= 0.8

    # this condition checks if even to attempt to utilize apsidal line symmetry approximations
    # curve has to have enough point on orbit and have to span at least in 0.8 phase
    try_to_find_appx = _look_for_approximation(phases_span_test, not self.has_pulsations())

    if try_to_find_appx:
        # in case of clean surface or synchronous rotation (more-less), symmetry around semi-major axis can be utilized
        # mask isolating the symmetrical part of the orbit
        azimuths = all_orbital_pos_arr[:, 2]

        # test whether mirroring around semi-major axis will be performed
        # todo: consider asynchronosu test
        approximation_test1 = _eval_approximation_one(self, phases)

        fn = lambda: prepare_geosymmetric_orbit(self, azimuths, phases)
        reduced_phase_ids, counterpart_postion_arr, reduced_phase_mask = fn()

        # spliting orbital motion into two separate groups on different sides of apsidal line
        reduced_orbit_arr, reduced_orbit_supplement_arr = \
            _split_orbit_by_apse_line(all_orbital_pos_arr, reduced_phase_mask)

        # replace code with method which will find supplements to each other

        # todo: unittest this method
        # if `index_of_closest` is applied on `reduced_orbit_supplement_arr` variable, you will get values which are
        # related to `reduced_orbit_arr`
        # example: reduced_orbit_supplement_arr[index_of_closest[idx]] related to reduced_orbit_arr[idx]
        index_of_closest = utils.find_idx_of_nearest(reduced_orbit_supplement_arr[:, 1], reduced_orbit_arr[:, 1])
        # testing whether all counterpart phases were assigned to template part of orbital motion
        # fixme: add outlier point to computational site like [point, None] or whatever like that???
        isin_test = np.isin(np.arange(np.count_nonzero(~reduced_phase_mask)), index_of_closest)
        # finding indices of reduced_orbit_supplement_arr which were not assigned to any symmetricall orbital position
        missing_phases_indices = np.arange(np.count_nonzero(~reduced_phase_mask))[~isin_test]

        # finding index of closest symmetrical orbital position to the missing phase
        index_of_closest_reversed = []
        if len(missing_phases_indices) > 0:
            index_of_closest_reversed = utils.find_idx_of_nearest(reduced_orbit_arr[:, 1],
                                                                  reduced_orbit_supplement_arr[missing_phases_indices, 1])
            index_of_closest = np.append(index_of_closest, missing_phases_indices)
            reduced_orbit_arr = np.append(reduced_orbit_arr, reduced_orbit_arr[index_of_closest_reversed], axis=0)

        forward_radii = self.calculate_all_forward_radii(reduced_orbit_arr[:, 1], components=None)
        # calculating change in forward radius as a indicator of change in overall geometry, not calculated for the
        # first OrbitalPosition since it is True
        forward_radii = np.array(list(forward_radii.values()))
        rel_d_radii = np.abs(forward_radii[:, 1:] - np.roll(forward_radii, shift=1, axis=1)[:, 1:]) / forward_radii[:, 1:]
        # second approximation does not interpolates the resulting light curve but assumes that geometry is the same as
        # the geometry of the found counterpart
        # testing if change in geometry will not be too severe, you should rather use changes in point radius instead
        forward_radii_sorted = np.sort(forward_radii, axis=1)
        rel_d_radii_sorted = np.abs(forward_radii_sorted - np.roll(forward_radii_sorted, shift=1, axis=1)) / \
                             forward_radii_sorted

        approximation_test2 = np.max(rel_d_radii_sorted[:, 1:]) < config.MAX_RELATIVE_D_R_POINT and \
                              self.primary.synchronicity == 1.0 and self.secondary.synchronicity == 1.0  # spots???

        # this part checks if differences between geometries of adjacent phases are small enough to assume that
        # geometries are the same.
        new_geometry_test = calculate_new_geometry(self, reduced_orbit_arr, rel_d_radii)

    else:
        approximation_test1 = False
        approximation_test2 = False

    # initial values of radii to be compared with
    # orig_forward_rad_p, orig_forward_rad_p = 100.0, 100.0  # 100.0 is too large value, it will always fail the first
    # test and therefore the surface will be built

    logger_messages = {
        'zero': 'lc will be calculated in a rigorous `phase to phase manner` without approximations',
        'one': 'one half of the points on LC on the one side of the apsidal line will be interpolated',
        'two': 'geometry of the stellar surface on one half of the apsidal '
               'line will be copied from their symmetrical counterparts'
    }

    all_orbital_pos = utils.convert_binary_orbital_motion_arr_to_positions(all_orbital_pos_arr)
    orbital_motion_counterpart = utils.convert_binary_orbital_motion_arr_to_positions(counterpart_postion_arr)
    if approximation_test1:
        __logger__.info('one half of the points on LC on the one side of the apsidal line will be interpolated')
        band_curves = integrate_lc_appx_one(self, all_orbital_pos, orbital_motion_counterpart, reduced_phase_ids,
                                            reduced_phase_mask, ecl_boundaries, phases,
                                            counterpart_postion_arr, new_geometry_test, **kwargs)

    elif approximation_test2:
        __logger__.info('geometry of the stellar surface on one half of the apsidal '
                        'line will be copied from their symmetrical counterparts')
        band_curves = integrate_lc_appx_two(self, all_orbital_pos, missing_phases_indices, index_of_closest,
                                            index_of_closest_reversed, reduced_phase_mask, ecl_boundaries, phases,
                                            new_geometry_test, **kwargs)

    else:
        __logger__.info('lc will be calculated in a rigorous phase to phase manner without approximations')
        band_curves = integrate_lc_exactly(self, all_orbital_pos, phases, ecl_boundaries, **kwargs)

    return band_curves






