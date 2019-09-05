import numpy as np


def get_flaten_properties(component):
    """
    Return flatten ndarrays of points, faces, etc. from object instance and spot instances for given object.

    :param component: Star instance
    :return: Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]

    ::

        Tuple(points, normals, faces, temperatures, log_g, rals, face_centres)
    """
    points = component.points
    normals = component.normals
    faces = component.faces
    temperatures = component.temperatures
    log_g = component.log_g
    rals = {mode_idx: mode.rals[0] for mode_idx, mode in component.pulsations.items()}
    centres = component.face_centres

    if isinstance(component.spots, (dict,)):
        for idx, spot in component.spots.items():
            faces = np.concatenate((faces, spot.faces + len(points)), axis=0)
            points = np.concatenate((points, spot.points), axis=0)
            normals = np.concatenate((normals, spot.normals), axis=0)
            temperatures = np.concatenate((temperatures, spot.temperatures), axis=0)
            log_g = np.concatenate((log_g, spot.log_g), axis=0)
            for mode_idx, mode in component.pulsations.items():
                rals[mode_idx] = np.concatenate((rals[mode_idx], mode.rals[1][idx]), axis=0)
            centres = np.concatenate((centres, spot.face_centres), axis=0)

    return points, normals, faces, temperatures, log_g, rals, centres


def potential_from_radius(self, component, radius, phi, theta, component_distance):
    """
    calculate potential given spherical coordinates radius, phi, theta

    :param self: BinarySystem
    :param component: 'primary` or `secondary`
    :param radius: float
    :param phi: float
    :param theta: float
    :param component_distance: float
    :return: float
    """
    precalc_fn = self.pre_calculate_for_potential_value_primary if component == 'primary' else \
        self.pre_calculate_for_potential_value_secondary
    potential_fn = self.potential_value_primary if component == 'primary' else \
        self.potential_value_secondary

    precalc_args = (component_distance, phi, theta)
    args = precalc_fn(*precalc_args)
    return potential_fn(radius, *args)


def calculate_phase(time, period, T0, offset=0.5):
    """
    calculates photometric phase from observations

    :param time: array
    :param period: array
    :param T0: float
    :param offset: float
    :return: array
    """
    return np.mod((time - T0 + offset * period) / period, 1.0) - offset
