from uuid import uuid4


def prepare_observe_args(orbital_motion):
    return ((idx, ) + position for idx, position in enumerate(orbital_motion))


def observe_worker(*args):
    class Self(object):
        def __init__(self):
            self.system = None
    self = Self()

    initial_sys_args, system_cls, xargs = args
    self.system = system_cls(name=str(uuid4()), **initial_sys_args)

    self.system.build_mesh(components_distance=1, suppress_parallelism=True)


