from abc import ABCMeta, abstractmethod
from astropy import units as u
import numpy as np

class Body(object, metaclass=ABCMeta):
    """
    Abstract class defining bodies that can be modelled by this software
    see https://docs.python.org/3.5/library/abc.html for more informations
    """

    __metaclass__ = ABCMeta

    def __init__(self, name=None):
        """
        parameters of abstract class Body except 'name' will be defined later independetly 
        after initialization of the instance None will be their default value
        name of the body is required, otherwise exception will rise
        """
        if name is None:
            raise NameError('Name of the body not found. Please name the body during initialization: xy = Body(name=\'Name\')')
        else:
            self._name = str(name)

        #initializing other parameters to None
        self._mass = None
        self._t_eff = None

        #setting default unit
        self._mass_default_unit = u.solMass
        self._temperature_default_unit = u.K


    #Getters and setters
    #____________________________________________________________________
    @property
    def name(self):
        """
        name getter
        call this by xy.name
        """
        return self._name

    @name.setter
    def name(self, name=None):
        """
        name setter
        call this by xy.name = new_name
        """
        self._name = name

    #..........................................................................................
    @property
    def mass(self):
        """
        mass getter
        call this by xy.mass
        """
        return self._mass

    @mass.setter
    def mass(self, mass=None):
        """
        mass setter
        call this by xy.mass = new_mass * unit
        you can use any mass unit, this method will make conversion to solar mass
        """
        if mass.unit is self._mass_default_unit:
            self._mass = mass
        else:
            self._mass = mass.unit.to(self._mass_default_unit, mass.value) * self._mass_default_unit

    @property
    def mass_default_unit(self):
        """
        mass default unit getter
        call this by xy.mass_default_unit
        """
        return self._mass_default_unit

    @mass_default_unit.setter
    def mass_default_unit(self, mass_default_unit='None'):
        """
        mass default unit setter
        call this by xy.mass_default_unit = new_mass_default_unit
        you can change it to any mass unit but we recommend to leave it set to solar mass
        make sure to use correct astropy.units notation,
        see http://docs.astropy.org/en/v0.2.1/units/index.html
        """

        if mass_default_unit is not None:
            mass_default_unit.to(u.solMass) #check if new default unit is unit of mass
            self._mass_default_unit = mass_default_unit

    #.........................................................................................
    @property
    def t_eff(self):
        """
        effective temperature getter
        call this by xy.t_eff
        """
        return self._t_eff

    @t_eff.setter
    def t_eff(self, t_eff=None):
        """
        effective temperature setter
        call this by xy.t_eff = new_t_eff * unit
        this function accepts only Kelvins, if your input is without unit, function assumes that value is in Kelvins
        """

        _tp = type(t_eff)
        if _tp is u.quantity.Quantity:
            if t_eff.unit is u.K:
                self._t_eff = t_eff
            else:
                raise TypeError('Function accepts only Kelvins or no units.')
        elif (_tp is int) or (_tp is float) or (_tp is np.int) or (_tp is np.float):
            self._t_eff = float(t_eff) * u.K
        else:
            raise TypeError('Make sure that input value is int, float or unit.Quantity object')

    # .........................................................................................
    @property
    def vertices(self):
        """
        vertices getter for Body
        call this by xy.t_eff
        :returns vertices of objects in format:

        """
        pass

    @vertices.setter
    def vertices(self, vertices='None'):
        """
        vertices setter for Body class


        :param vertices:
        :return:
        """
        pass



    #..............................................................................................

