from engine.system import System

class BinarySystem(System):

    def __init__(self, inclination=None, name=None):
        super(BinarySystem, self).__init__(name=name)
        self._inclination = inclination

    @property
    def inclination(self):
        return self._inclination
