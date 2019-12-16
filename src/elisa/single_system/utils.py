def move_sys_onpos(system, orbital_position, primary_potential=None, secondary_potential=None, on_copy=True):
    """
    Prepares a postion container for given orbital position.
    Supplied `system` is not affected if `on_copy` is set to True.

    Following methods are applied::

        system.set_on_position_params()
        system.flatt_it()
        system.apply_rotation()
        system.apply_darkside_filter()

    :param system: elisa.single_system.container.PositionContainer;
    :param orbital_position: collections.namedtuple; elisa.const.SinglePosition;
    :return: container; elisa.binary_system.container.PositionContainer;
    :param primary_potential: float;
    :param secondary_potential: float;
    :param on_copy: bool;
    """
    if on_copy:
        system = system.copy()
    system.set_on_position_params(orbital_position, primary_potential, secondary_potential)
    system.flatt_it()
    system.apply_rotation()
    system.apply_darkside_filter()
    return system
