import numpy as np


def get_flaten_properties(component):
    points = component.points
    normals = component.normals
    faces = component.faces
    temperatures = component.temperatures
    log_g = component.log_g

    if isinstance(component.spots, (dict,)):
        for idx, spot in component.spots.items():
            faces = np.concatenate((faces, spot.faces + len(points)), axis=0)
            points = np.concatenate((points, spot.points), axis=0)
            normals = np.concatenate((normals, spot.normals), axis=0)
            temperatures = np.concatenate((temperatures, spot.temperatures), axis=0)
            log_g = np.concatenate((log_g, spot.log_g), axis=0)

    return points, normals, faces, temperatures, log_g
