from astropy import units as u
from engine import units
import numpy as np
import logging
from engine import utils

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')

class Spot(object):
    KWARGS = ["longitude", "latitude", "angular_density", "angular_diameter", "temperature_factor"]

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs=kwargs, kwarglist=Spot.KWARGS, instance=Spot)

        self._angular_density = None
        self._latitude = None
        self._longitude = None
        self._angular_diameter = None
        self._temperature_factor = None

        self._logger = logging.getLogger(Spot.__name__)

    @property
    def longitude(self):
        return self._longitude

    @longitude.setter
    def longitude(self, longitude):
        """
        setter for spot longitude
        expecting value in degrees or as astropy units instance

        :param longitude: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(longitude, u.quantity.Quantity):
            self._longitude = np.float64(longitude.to(units.ARC_UNIT))
        elif isinstance(longitude, (int, np.int, float, np.float)):
            self._longitude = np.float64(longitude)
        else:
            raise TypeError('Input of variable `longitude` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def latitude(self):
        return self._latitude

    @latitude.setter
    def latitude(self, latitude):
        """
        setter for spot latitude
        expecting value in degrees or as astropy units instance

        :param latitude: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(latitude, u.quantity.Quantity):
            self._longitude = np.float64(latitude.to(units.ARC_UNIT))
        elif isinstance(latitude, (int, np.int, float, np.float)):
            self._longitude = np.float64(latitude)
        else:
            raise TypeError('Input of variable `latitude` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def angular_diamter(self):
        return self._angular_diameter

    @angular_diamter.setter
    def angular_diamter(self, angular_diamter):
        """
        setter for spot angular_diamter
        expecting value in degrees or as astropy units instance

        :param angular_diamter: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(angular_diamter, u.quantity.Quantity):
            self._longitude = np.float64(angular_diamter.to(units.ARC_UNIT))
        elif isinstance(angular_diamter, (int, np.int, float, np.float)):
            self._longitude = np.float64(angular_diamter)
        else:
            raise TypeError('Input of variable `angular_diamter` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def angular_density(self):
        return self._angular_density

    @angular_density.setter
    def angular_density(self, angular_density):
        """
        setter for spot angular_density
        expecting value in degrees or as astropy units instance

        :param angular_density: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(angular_density, u.quantity.Quantity):
            self._longitude = np.float64(angular_density.to(units.ARC_UNIT))
        elif isinstance(angular_density, (int, np.int, float, np.float)):
            self._longitude = np.float64(angular_density)
        else:
            raise TypeError('Input of variable `angular_density` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def temperature_factor(self):
        return self._temperature_factor

    @temperature_factor.setter
    def temperature_factor(self, temperature_factor):
        """
       setter for spot temperature_factor

       :param temperature_factor: (np.)int, (np.)float
       :return:
       """
        if isinstance(temperature_factor, (int, np.int, float, np.float)):
            self._longitude = np.float64(temperature_factor)
        else:
            raise TypeError('Input of variable `temperature_factor` is not (np.)int or (np.)float.')
