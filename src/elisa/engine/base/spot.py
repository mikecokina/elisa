import numpy as np

from astropy import units as u
from elisa.engine import units, logger
from elisa.engine import utils
from elisa.engine.utils import is_empty


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

        self._discretization_factor = np.nan
        self._latitude = np.nan
        self._longitude = np.nan
        self._angular_diameter = np.nan
        self._temperature_factor = np.nan

        self.boundary = np.array([])
        self.boundary_center = np.array([])
        self.center = np.array([])

        self.points = np.array([])
        self.normals = np.array([])
        self.faces = np.array([])
        self.face_centres = np.array([])

        self.areas = np.array([])
        self.potential_gradient_magnitudes = np.array([])
        self.temperatures = np.array([])

        self._log_g = np.array([])

        for key in kwargs:
            set_val = kwargs.get(key)
            self._logger.debug(f"setting property {key} of class instance {self.__class__.__name__} to {kwargs[key]}")
            setattr(self, key, set_val)

    def kwargs_serializer(self):
        """
        Serializer and return mandatory kwargs of sefl (Spot) instance to dict.

        :return: Dict

        ::

            { kwarg: value }
        """
        return {kwarg: getattr(self, kwarg) for kwarg in self.MANDATORY_KWARGS if not is_empty(getattr(self, kwarg))}

    @property
    def log_g(self):
        """
        :return: ndarray
        """
        return self._log_g

    @log_g.setter
    def log_g(self, log_g):
        """
        :param log_g: ndarray
        :return:
        """
        self._log_g = log_g

    @property
    def longitude(self):
        """
        :return: float
        """
        return self._longitude

    @longitude.setter
    def longitude(self, longitude):
        """
        Expecting value in degrees or as astropy units instance.

        :param longitude: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(longitude, u.quantity.Quantity):
            self._longitude = np.float64(longitude.to(units.ARC_UNIT))
        elif isinstance(longitude, (int, np.int, float, np.float)):
            self._longitude = np.radians(np.float64(longitude))
        else:
            raise TypeError('Input of variable `longitude` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def latitude(self):
        """
        :return: float
        """
        return self._latitude

    @latitude.setter
    def latitude(self, latitude):
        """
        Expecting value in degrees or as astropy units instance.

        :param latitude: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(latitude, u.quantity.Quantity):
            self._latitude = np.float64(latitude.to(units.ARC_UNIT))
        elif isinstance(latitude, (int, np.int, float, np.float)):
            self._latitude = np.radians(np.float64(latitude))
        else:
            raise TypeError('Input of variable `latitude` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def angular_diameter(self):
        """
        :return: float
        """
        return self._angular_diameter

    @angular_diameter.setter
    def angular_diameter(self, angular_diameter):
        """
        Expecting value in degrees or as astropy units instance.

        :param angular_diamter: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(angular_diameter, u.quantity.Quantity):
            self._angular_diameter = np.float64(angular_diameter.to(units.ARC_UNIT))
        elif isinstance(angular_diameter, (int, np.int, float, np.float)):
            self._angular_diameter = np.radians(np.float64(angular_diameter))
        else:
            raise TypeError('Input of variable `angular_diamter` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def discretization_factor(self):
        """
        :return: float
        """
        return self._discretization_factor

    @discretization_factor.setter
    def discretization_factor(self, discretization_factor):
        """
        Setter for spot discretization_factor (mean angular size of spot face)
        Expecting value in degrees or as astropy units instance.

        :param discretization_factor: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(discretization_factor, u.quantity.Quantity):
            self._discretization_factor = np.float64(discretization_factor.to(units.ARC_UNIT))
        elif isinstance(discretization_factor, (int, np.int, float, np.float)):
            self._discretization_factor = np.radians(np.float64(discretization_factor))
        else:
            raise TypeError('Input of variable `angular_density` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def temperature_factor(self):
        return self._temperature_factor

    @temperature_factor.setter
    def temperature_factor(self, temperature_factor):
        """
       :param temperature_factor: (numpy.)int, (numpy.)float
       :return:
       """
        if isinstance(temperature_factor, (int, np.int, float, np.float)):
            self._temperature_factor = np.float64(temperature_factor)
        else:
            raise TypeError('Input of variable `temperature_factor` is not (numpy.)int or (numpy.)float.')

    def calculate_areas(self):
        """
        Returns areas of each face of the spot build_surface.

        :return: ndarray:

        ::

            numpy.array([area_1, ..., area_n])
        """
        return utils.triangle_areas(triangles=self.faces, points=self.points)

