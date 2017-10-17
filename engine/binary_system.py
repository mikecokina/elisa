from engine.system import System


class BinarySystem(System):

    def __init__(self, inclination=None, name=None):
        super(BinarySystem, self).__init__(name=name)
        self._inclination = inclination

    @property
    def inclination(self):
        """
        inclination of binary star system
        :return:
        """
        return self._inclination

    @inclination.setter
    def inclination(self, inclination):
        self._inclination = inclination

    def compute_lc(self):
        pass

    def get_info(self):
        pass
