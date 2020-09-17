def eval_args_for_magnitude_gradient(star_container):
    """
    returns either all of the surface pints (not spots) in case of the surface with spots or just its symmetrical part
    in case of clean surface

    :param star_container: StarContainer;
    :return: tuple;
    """
    if star_container.symmetry_test():
        points = star_container.points[:star_container.base_symmetry_points_number]
        faces = star_container.faces[:star_container.base_symmetry_faces_number]
    else:
        points, faces = star_container.points, star_container.faces
    return points, faces
