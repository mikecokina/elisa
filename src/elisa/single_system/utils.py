from elisa import utils


def move_sys_onpos(system, on_copy=True):
    """
    Prepares a postion container for given orbital position.
    Supplied `system` is not affected if `on_copy` is set to True.

    Following methods are applied::

        system.set_on_position_params()
        system.flatt_it()
        system.apply_rotation()
        system.apply_darkside_filter()

    :param system: elisa.single_system.container.PositionContainer;
    :param on_copy: bool;
    :return: container; elisa.sinary_system.container.PositionContainer;
    """
    if on_copy:
        system = system.copy()
    system.flatt_it(system_container=system, components=['star'])
    system = utils.apply_rotation(system_container=system, components=['star'])
    system = utils.apply_darkside_filter(system_container=system, components=['star'])
    return system
