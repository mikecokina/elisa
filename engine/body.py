from abc import ABCMeta, abstractmethod
from astropy import units as u
import numpy as np
import logging
from engine import units as U


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class Body(object, metaclass=ABCMeta):
    """
    Abstract class defining bodies that can be modelled by this software
    see https://docs.python.org/3.5/library/abc.html for more informations
    units are imported from astropy.units module
    see documentation http://docs.astropy.org/en/stable/units/
    """
    __metaclass__ = ABCMeta

    ID = 1
    KWARGS = []

    def __init__(self, name=None, **kwargs):
        """
        Parameters of abstract class Body
        """
        self._logger = logging.getLogger(Body.__name__)

        if name is None:
            self._name = str(Body.ID)
            self._logger.debug("Name of class instance {} set to {}".format(Body.__name__, self._name))
            Body.ID += 1
        else:
            self._name = str(name)

        # initializing other parameters to None
        self._mass = None  # float64
        self._t_eff = None  # float64
        self._points = None  # numpy.array of float64
        self._faces = None  # dict
        self._normals = None  # dict
        self._temperatures = None  # dict
        # self._intensity = None # dict
        self._synchronicity = None  # float64
        self._albedo = None  # float64
        self._polar_radius = None  # float64
        self._areas = None
        self._discretization_factor = 3

        # values of properties
        # toto tu uz byt nemusi?
        for kwarg in self.KWARGS:
            if kwarg in kwargs:
                setattr(self, kwarg, kwargs[kwarg])

    # Getters and setters
    @property
    def name(self):
        """
        name getter
        usage: xy.name

        :return: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        name setter
        usage: xy.name = new_name

        :param name: str
        """
        self._name = str(name)

    @property
    def mass(self):
        """
        mass getter, returns mass of object in default mass unit
        usage: by xy.mass

        :return: np.float64
        """
        # return self._mass * self.__MASS_UNIT.to(u.solMass) * u.solMass
        return self._mass

    @mass.setter
    def mass(self, mass):
        """
        mass setter
        usage: xy.mass = new_mass
        if mass is int, np.int, float, np.float, program assumes solar mass as it's unit
        if mass astropy.unit.quantity.Quantity instance, program converts it to default units and stores it's value in
        attribute _mass

        :param mass: int, np.int, float, np.float, astropy.unit.quantity.Quantity
        """
        if isinstance(mass, u.quantity.Quantity):
            self._mass = np.float64(mass.to(U.MASS_UNIT))
        elif isinstance(mass, (int, np.int, float, np.float)):
            self._mass = np.float64(mass * u.solMass.to(U.MASS_UNIT))
        else:
            raise TypeError('Your input is not (np.)int or (np.)float nor astropy.unit.quantity.Quantity instance.')

    @property
    def t_eff(self):
        """
        effective temperature getter
        usage: xy.t_eff

        :return: numpy.float64
        """
        return self._t_eff

    @t_eff.setter
    def t_eff(self, t_eff):
        """
        effective temperature setter
        usage: xy.t_eff = new_t_eff
        this function accepts value in any temperature unit, if your input is without unit, function assumes that value
        is in Kelvins

        :param t_eff: int, np.int, float, np.float, astropy.unit.quantity.Quantity
        """
        if isinstance(t_eff, u.quantity.Quantity):
            self._t_eff = np.float64(t_eff.to(U.TEMPERATURE_UNIT))
        elif isinstance(t_eff, (int, np.int, float, np.float)):
            self._t_eff = np.float64(t_eff)
        else:
            raise TypeError('Value of `t_eff` is not (np.)int or (np.)float nor astropy.unit.quantity.Quantity '
                            'instance.')

    @property
    def points(self):
        """
        points getter
        usage: xy.points
        returns dictionary of points that forms surface of Body

        :return: dict
        """
        return self._points

    @points.setter
    def points(self, points):
        """
        points setter
        usage: xy.points = new_points
        setting numpy array of points that form surface of Body
        input dictionary has to be in shape:
        points = numpy.array([[x1 y1 z1],
                                [x2 y2 z2],
                                ...
                                [xN yN zN]])
        where xi, yi, zi are cartesian coordinates of vertice i

        :param points: numpy.array
        xi, yi, zi: float64
        """
        self._points = np.array(points)

    @property
    def faces(self):
        """
        returns dictionary of triangles that will create surface of body
        triangles are stored as list of indices of points
        usage: xy.faces

        :return: numpy.array
        shape: points = numpy.array([[vertice_index_k, vertice_index_l, vertice_index_m]),
                                  [...]),
                                   ...
                                  [...]])
        """
        return self._faces

    @faces.setter
    def faces(self, faces):
        """
        faces setter
        usage: xy.faces = new_faces
        faces dictionary has to be in shape:
        points = np.array([vertice_index_k, vertice_index_l, vertice_index_m],
                          [...],
                           ...
                          [...]]

        :param faces: numpy.array
        """
        self._faces = faces

    @property
    def normals(self):
        """
        returns array containing normalised outward facing normals of corresponding faces with same index
        usage: xy.normals

        :return: numpy.array
        shape: normals = numpy_array([[normal_x1, normal_y1, normal_z1],
                                      [normal_x2, normal_y2, normal_z2],
                                       ...
                                      [normal_xn, normal_yn, normal_zn]]
        """
        return self._normals

    @normals.setter
    def normals(self, normals):
        """
        setter for normalised outward facing normals of corresponding faces with same index
        usage: xy.normals = new_normals
        expected shape of normals matrix:
        normals = numpy_array([[normal_x1, normal_y1, normal_z1],
                               [normal_x2, normal_y2, normal_z2],
                                       ...
                               [normal_xn, normal_yn, normal_zn]]

        :param normals: numpy.array
        """
        self._normals = normals

    @property
    def areas(self):
        """
        returns array of areas of corresponding faces
        usage: xy.areas

        :return: np.array([area_1, ..., area_n])
        """
        return self._areas

    @areas.setter
    def areas(self, areas):
        """
        returns array of areas of corresponding faces
        usage: xy.areas = new_areas

        :param areas: numpy.array([area_1, ..., area_n])
        :return:
        """
        self._areas = areas

    @property
    def temperatures(self):
        """
        returns array of temeratures of corresponding faces
        usage: xy.temperatures

        :return:numpy.arrays
        shape: numpy.array([t_eff1, ..., t_effn])
        """
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures):
        """
        temperatures setter
        usage: xy.temperatures = new_temperatures
        setter for array of temeratures of corresponding faces in shape
        :shape: numpy.array([t_eff1, ..., t_effn])

        :param temperatures: numpy.array
        """
        self._temperatures = temperatures

    @property
    def synchronicity(self):
        """
        returns synchronicity parameter F = omega_rot/omega_orb
        usage: xy.synchronicity

        :return: numpy.float64
        """
        return self._synchronicity

    @synchronicity.setter
    def synchronicity(self, synchronicity):
        """
        object synchronicity (F = omega_rot/omega_orb) setter, expects number input convertible to numpy float64
        usage: xy.synchronicity = new_synchronicity

        :param synchronicity: numpy.float64
        """
        if synchronicity is not None:
            self._synchronicity = np.float64(synchronicity)
        else:
            self._synchronicity = None

    @property
    def albedo(self):
        """
        returns bolometric albedo of an object (reradiated energy/ irradiance energy)
        usage: xy.albedo

        :return: float64
        """
        return self._albedo

    @albedo.setter
    def albedo(self, albedo):
        """
        setter for bolometric albedo (reradiated energy/ irradiance energy)
        accepts value of albedo in range (0,1)
        usage xy.albedo = new_albedo

        :param albedo: float64
        """
        if 0 <= albedo <= 1:
            self._albedo = np.float64(albedo)
        else:
            raise ValueError('Parameter albedo = {} is out of range (0, 1)'.format(albedo))

    @property
    def polar_radius(self):
        """
        returns value polar radius of an object in default unit
        usage: xy.polar_radius

        :return: float64
        """
        return self._polar_radius

    @polar_radius.setter
    def polar_radius(self, polar_radius):
        """
        setter for polar radius of body
        expected type is astropy.units.quantity.Quantity, numpy.float or numpy.int othervise TypeError will be raised
        if quantity is not specified, default distance unit is assumed

        :param polar_radius:
        :return:
        """
        if isinstance(polar_radius, u.quantity.Quantity):
            self._polar_radius = np.float64(polar_radius.to(U.DISTANCE_UNIT))
        elif isinstance(polar_radius, (int, np.int, float, np.float)):
            self._polar_radius = np.float64(polar_radius)
        else:
            raise TypeError('Value of variable `polar radius` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def discretization_factor(self):
        """
        returns mean angular distance between surface points

        :return: float
        """
        return self._discretization_factor

    @discretization_factor.setter
    def discretization_factor(self, discretization_factor):
        """
        setter for discretization factor

        :param :float or int
        :return:
        """
        self._discretization_factor = discretization_factor

    @property
    def mass_unit(self):
        """
        returns default mass unit
        usage: xy.mass_unit

        :return: astropy.unit.quantity.Quantity
        """
        return U.MASS_UNIT

    @property
    def temperature_unit(self):
        """
        returns default unit of temperature
        usage: xy.temperature_unit

        :return: astropy.unit.quantity.Quantity
        """
        return U.TEMPERATURE_UNIT

    @property
    def distance_unit(self):
        """
        returns default unit of length
        usage: xy.length_unit

        :return: astropy.unit.quantity.Quantity
        """
        return U.DISTANCE_UNIT

    @property
    def time_unit(self):
        """
        returns default unit of time
        usage: xy.time_unit

        :return: astropy.unit.quantity.Quantity
        """
        return U.TIME_UNIT

    @property
    def arc_unit(self):
        """
        returns default unit of time
        usage: xy.arc_unit

        :return: astropy.unit.quantity.Quantity
        """
        return U.ARC_UNIT

    def calculate_normals(self, points, faces):
        """
        returns outward facing normal unit vector for each face of stellar surface

        :return: numpy_array([[normal_x1, normal_y1, normal_z1],
                              [normal_x2, normal_y2, normal_z2],
                               ...
                              [normal_xn, normal_yn, normal_zn]])
        """
        normals = np.array([np.cross(points[xx[1]] - points[xx[0]], points[xx[2]]
                                     - points[xx[0]]) for xx in faces])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        centres = self.calculate_surface_centres(points, faces)
        # possible problem is that this approach fails in case when triangle is perpendicular to x axis
        # fixme: so? then fix me
        sgn_vector = 0.5 * np.sum(np.sign(centres[:, 1:]) * np.sign(normals[:, 1:]), axis=1)

        return normals * sgn_vector[:, None]

    @staticmethod
    def calculate_surface_centres(points=None, faces=None):
        """
        returns centers of every surface face

        :return: numpy_array([[center_x1, center_y1, center_z1],
                              [center_x2, center_y2, center_z2],
                               ...
                              [center_xn, center_yn, center_zn]])
        """
        return np.average(points[faces], axis=1)

    def calculate_areas(self):
        """
        returns areas of each face of the star surface
        :return: numpy.array([area_1, ..., area_n])
        """
        return 0.5 * np.linalg.norm(np.cross(self.points[self.faces[:, 1]] - self.points[self.faces[:, 0]],
                                             self.points[self.faces[:, 2]] - self.points[self.faces[:, 0]]),
                                    axis=1)

    def get_info(self):
        pass
