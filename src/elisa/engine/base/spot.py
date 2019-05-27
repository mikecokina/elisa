import numpy as np

from astropy import units as u
from elisa.engine import units, logger
from elisa.engine import utils


class Spot(object):
    """
    Spot data container
    """
    MANDATORY_KWARGS = ["longitude", "latitude", "angular_diameter", "temperature_factor"]
    OPTIONAL_KWARGS = ["angular_density"]
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, suppress_logger=False, **kwargs):
        utils.invalid_kwarg_checker(kwargs=kwargs, kwarglist=Spot.ALL_KWARGS, instance=Spot)
        utils.check_missing_kwargs(Spot.MANDATORY_KWARGS, kwargs, instance_of=Spot)
        self._logger = logger.getLogger(Spot.__name__, suppress=suppress_logger)

        self._discretization_factor = None
        self._latitude = None
        self._longitude = None
        self._angular_diameter = None
        self._temperature_factor = None

        self.boundary = None
        self.boundary_center = None
        self.center = None

        self.points = None
        self.normals = None
        self.faces = None
        self.face_centres = None

        self.areas = None
        self.potential_gradient_magnitudes = None
        self.temperatures = None

        self._log_g = None

        for key in kwargs:
            set_val = kwargs.get(key)
            self._logger.debug(f"setting property {key} of class instance {self.__class__.__name__} to {kwargs[key]}")
            setattr(self, key, set_val)

    def kwargs_serializer(self):
        return {kwarg: getattr(self, kwarg) for kwarg in self.MANDATORY_KWARGS if getattr(self, kwarg) is not None}

    @property
    def log_g(self):
        return self._log_g

    @log_g.setter
    def log_g(self, log_g):
        self._log_g = log_g

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
            self._longitude = np.radians(np.float64(longitude))
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
            self._latitude = np.float64(latitude.to(units.ARC_UNIT))
        elif isinstance(latitude, (int, np.int, float, np.float)):
            self._latitude = np.radians(np.float64(latitude))
        else:
            raise TypeError('Input of variable `latitude` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def angular_diameter(self):
        return self._angular_diameter

    @angular_diameter.setter
    def angular_diameter(self, angular_diameter):
        """
        setter for spot angular_diamter
        expecting value in degrees or as astropy units instance

        :param angular_diamter: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(angular_diameter, u.quantity.Quantity):
            self._angular_diameter = np.float64(angular_diameter.to(units.ARC_UNIT))
        elif isinstance(angular_diameter, (int, np.int, float, np.float)):
            self._angular_diameter = np.radians(np.float64(angular_diameter))
        else:
            raise TypeError('Input of variable `angular_diamter` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def discretization_factor(self):
        return self._discretization_factor

    @discretization_factor.setter
    def discretization_factor(self, discretization_factor):
        """
        setter for spot discretization_factor (mean angular size of spot face)
        expecting value in degrees or as astropy units instance

        :param discretization_factor: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(discretization_factor, u.quantity.Quantity):
            self._discretization_factor = np.float64(discretization_factor.to(units.ARC_UNIT))
        elif isinstance(discretization_factor, (int, np.int, float, np.float)):
            self._discretization_factor = np.radians(np.float64(discretization_factor))
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
            self._temperature_factor = np.float64(temperature_factor)
        else:
            raise TypeError('Input of variable `temperature_factor` is not (np.)int or (np.)float.')

    def calculate_areas(self):
        """
        returns areas of each face of the spot build_surface
        :return: numpy.array([area_1, ..., area_n])
        """
        return utils.triangle_areas(triangles=self.faces, points=self.points)

