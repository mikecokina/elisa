import numpy as np

from elisa.engine import const, utils


def get_critical_inclination(binary, components_distance: float):
    if binary.morphology != 'over-contact':
        radius1 = np.mean([binary.primary.side_radius, binary.primary.forward_radius, binary.primary.backward_radius,
                           binary.primary.polar_radius])
        radius2 = np.mean([binary.secondary.side_radius, binary.secondary.forward_radius,
                           binary.secondary.backward_radius, binary.secondary.polar_radius])
        cos_i_critical = (radius1 + radius2) / components_distance
        return np.degrees(np.arccos(cos_i_critical))


def get_eclipse_boundaries(binary, components_distance: float):
    # check whether the inclination is high enough to enable eclipses
    if binary.morphology != 'over-contact':
        radius1 = np.mean([binary.primary.side_radius, binary.primary.forward_radius, binary.primary.backward_radius,
                           binary.primary.polar_radius])
        radius2 = np.mean([binary.secondary.side_radius, binary.secondary.forward_radius,
                           binary.secondary.backward_radius, binary.secondary.polar_radius])
        sin_i_critical = (radius1 + radius2) / components_distance
        sin_i = np.sin(binary.inclination)
        if sin_i < sin_i_critical:
            binary._logger.debug('Inclination is not sufficient to produce eclipses.')
            return None
        radius1 = binary.primary.forward_radius
        radius2 = binary.secondary.forward_radius
        sin_i_critical = (radius1 + radius2) / components_distance
        azimuth = np.arcsin(np.sqrt(np.power(sin_i_critical, 2) - np.power(np.cos(binary.inclination), 2)))
        azimuths = np.array([const.FULL_ARC - azimuth, azimuth, const.PI - azimuth, const.PI + azimuth])
        return azimuths


def darkside_filter(sight_of_view, *args, **kwargs):
    pass


# def sight_of_view_by_orbital_position(inclination: float, orbital_motion: list):
#     orbital_motion = np.array(orbital_motion)
#     inclination_reference_vector = np.array([0.0, 1.0, 0.0])
#     sov_reference_vector = np.array([1.0, 0.0, 0.0])
#     inclination_rotation_angle = const.PI - inclination
#
#     reverse_sov_back_inclinated = utils.arbitrary_rotation(theta=-inclination_rotation_angle,
#                                                            omega=inclination_reference_vector,
#                                                            vector=sov_reference_vector,
#                                                            degrees=False)
#     reverse_sov = np.array([utils.arbitrary_rotation(
#         theta=-azimuth,
#         omega=[0.0, 0.0, 1.0],
#         vector=reverse_sov_back_inclinated,
#         degrees=False
#     )
#         for azimuth in orbital_motion.T[2]
#     ])
#     sov = -1 * reverse_sov
#
#     # sov_xyz_coo = np.array([minus_sov_xy_coo[0] * -1, minus_sov_xy_coo[1] * -1, [0.0] * len(minus_sov_xy_coo[0])])
#     # sov_inclinated_coo = [utils.arbitrary_rotation(theta=inclination,
#     #                                                omega=[sov[0], -sov[1], 0.0],
#     #                                                vector=sov,
#     #                                                degrees=False) for sov in sov_xyz_coo.T]
#     #
#     #
#     # print(sov_inclinated_coo)
#     #
#     #
#
#     from matplotlib import pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(sov.T[0], sov.T[1], sov.T[2])
#     plt.show()
#
#
#     pass
