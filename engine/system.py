from abc import ABCMeta, abstractmethod


class System(object, metaclass=ABCMeta):
    """
    Abstract class defining System
    see https://docs.python.org/3.5/library/abc.html for more infromations
    """

    __metaclass__ = ABCMeta

    ID = 1

    def __init__(self, name=None):
        if name is None:
            self._name = str(System.ID)
            System.ID += 1
        else:
            self._name = str(name)

        # Default values of properties
        self._gamma = None

    @property
    def name(self):
        """
        name of object initialized on base of this abstract class
        :return: str
        """
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def gamma(self):
        """
        system center of mass radial velocity
        :return:
        """
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @abstractmethod
    def compute_lc(self):
        pass

    @abstractmethod
    def get_info(self):
        pass

