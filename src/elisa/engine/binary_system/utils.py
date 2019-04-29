import numpy as np


def get_flaten_properties(component):
    points = component.points
    normals = component.normals
    faces = component.faces
    temperatures = component.temperatures
    log_g = component.log_g

    points_index_map = np.array([(-1, i) for i in range(len(points))])
    normals_index_map = np.array([(-1, i) for i in range(len(normals))])

    if isinstance(component.spots, (dict,)):
        for idx, spot in component.spots.items():
            faces = np.concatenate((faces, spot.faces + len(points)), axis=0)
            points = np.concatenate((points, spot.points), axis=0)
            normals = np.concatenate((normals, spot.normals), axis=0)
            temperatures = np.concatenate((temperatures, spot.temperatures), axis=0)
            log_g = np.concatenate((log_g, spot.log_g), axis=0)

            p, n, = len(points_index_map), len(normals_index_map)

            points_index_map = np.concatenate(
                (points_index_map,
                 np.array([(idx, i) for i in range(p, p + len(spot.points))])),
                axis=0)
            normals_index_map = np.concatenate(
                (normals_index_map,
                 np.array([(idx, i) for i in range(n, n + len(spot.normals))])),
                axis=0)
    return points, normals, faces, temperatures, log_g, points_index_map, normals_index_map
