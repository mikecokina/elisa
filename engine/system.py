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

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @abstractmethod
    def compute_lc(self):
        pass

