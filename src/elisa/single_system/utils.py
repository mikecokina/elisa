from .. import const, utils


def move_sys_onpos(system, position, on_copy=True):
    """
    Prepares a postion container for given orbital position.
    Supplied `system` is not affected if `on_copy` is set to True.

    Following methods are applied::

        system.set_on_position_params()
        system.flatt_it()
        system.apply_rotation()
        system.apply_darkside_filter()

    :param position: collections.namedtuple; elisa.const.SinglePosition;
    :param system: elisa.single_system.container.PositionContainer;
    :param on_copy: bool;
    :return: container; elisa.sinary_system.container.PositionContainer;
    """
    if on_copy:
        system = system.copy()
    system.set_on_position_params(position)
    system.flatt_it()
    system.apply_rotation()
    system.add_secular_velocity()
    system.calculate_face_angles(line_of_sight=const.LINE_OF_SIGHT)
    system.apply_darkside_filter()
    return system


def calculate_volume(system):
    """
    Returns volume of rotationally squashed star based on a volume of elipsoid.

    :param system: elisa.SingleSystem;
    :return: float
    """
    return utils.calculate_ellipsoid_volume(system.star.polar_radius, system.star.equatorial_radius,
                                            system.star.equatorial_radius)
