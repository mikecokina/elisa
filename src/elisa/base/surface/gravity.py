def eval_args_for_magnitude_gradient(star_container):
    """
    returns either all of the surface pints (not spots) in case of the surface with spots or just its symmetrical part
    in case of clean surface

    :param star_container: StarContainer;
    :return: Tuple;
    """
    if star_container.symmetry_test():
        points = star_container.symmetry_points()
        faces = star_container.symmetry_faces(star_container.faces)
    else:
        points, faces = star_container.points, star_container.faces
    return points, faces
