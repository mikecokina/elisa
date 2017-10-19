from abc import ABCMeta, abstractmethod
from astropy import units as u
import numpy as np


class Body(object, metaclass=ABCMeta):
    """
    Abstract class defining bodies that can be modelled by this software
    see https://docs.python.org/3.5/library/abc.html for more informations
    internal units are imported from astropy.units module
    see documentation http://docs.astropy.org/en/stable/units/
    """
    __metaclass__ = ABCMeta

    ID = 1

    def __init__(self, name=None):
        """
        Parameters of abstract class Body
        """
        if name is None:
            self._name = str(Body.ID)
            Body.ID += 1
        else:
            self._name = str(name)

        # initializing other parameters to None
        self._mass = None  # float64
        self._t_eff = None  # float64
        self._vertices = None  # dict
        self._faces = None  # dict
        self._normals = None  # dict
        self._temperatures = None  # dict

        # setting default unit
        self._mass_unit = u.kg
        self._temperature_unit = u.K
        self._distance_unit = u.m
        self._time_unit = u.s

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
        mass getter, returns mass of object in solar masses
        usage: by xy.mass

        :return: np.float64
        """
        # return self._mass * self._mass_unit.to(u.solMass) * u.solMass
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
            self._mass = np.float64(mass.to(self._mass_unit))
        elif isinstance(mass, (int, np.int, float, np.float)):
            self._mass = np.float64(mass * u.solMass.to(u.kg))
        else:
            raise TypeError('Your input is not (np.)int or (np.)float nor astropy.unit.quantity.Quantity instance.')

    # podla mna je pouzivatela chuj po tom jaku jednotku my interne pouzivame
    # @mass_unit.setter
    # def mass_unit(self, mass_unit):
    #     """
    #     mass default unit setter
    #     call this by xy.mass_default_unit = new_mass_default_unit
    #     you can change it to any mass unit but we recommend to leave it set to current value
    #     make sure to use correct astropy.units notation,
    #     see http://docs.astropy.org/en/v0.2.1/units/index.html
    #
    #     :param mass_unit: astropy.unit.quantity.Quantity
    #     """
    #
    #     mass_unit.to(self._mass_unit)  # check if new default unit is unit of mass
    #     self._mass_unit = mass_unit

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
        this function accepts only Kelvins, if your input is without unit, function assumes that value is in Kelvins

        :param t_eff: int, np.int, float, np.float, astropy.unit.quantity.Quantity
        """
        if isinstance(t_eff, u.quantity.Quantity):
            self._t_eff = np.float64(t_eff.to(self._temperature_unit))
        elif isinstance(t_eff, (int, np.int, float, np.float)):
            self._t_eff = np.float64(t_eff)
        else:
            raise TypeError('Your input is not (np.)int or (np.)float nor astropy.unit.quantity.Quantity instance.')

    @property
    def vertices(self):
        """
        vertices getter
        usage: xy.vertices
        returns dictionary of points that forms surface of Body

        :return: dict
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        """
        vertices setter
        usage: xy.vertices = new_vertices
        setting dictionary of points that form surface of Body
        input dictionary has to be in shape:
        vertices = {vertice_ID_1: np.array([x1, y1, z1]),
                    vertice_ID_2: np.array([x2, y2, z2]),
                    ...
                    vertice_ID_N: np.array([xN, yN, zN])}
        where vertice_name_i is unique integer ID of vertice
        and xi, yi, zi are cartesian coordinates of vertice i


        :param vertices: dict
        vertice_name_i: int?
        xi, yi, zi: float64
        """
        self._vertices = {np.uint32(xx): np.array([np.float64(yy) for yy in vertices[xx]]) for xx in vertices}

    @property
    def faces(self):
        """
        returns dictionary of triangles that will create surface of body
        triangles are stored as list of keys of vertices
        usage: xy.faces

        :return: dict
        shape: vertices = {face_ID_1: np.array([vertice_ID_k, vertice_ID_l, vertice_ID_m]),
                           face_ID_2: np.array([...]),
                           ...
                           face_ID_n: np.array([...])}
        """
        return self._faces

    @faces.setter
    def faces(self, faces):
        """
        faces setter
        usage: xy.faces = new_faces
        faces dictionary has to be in shape:
        vertices = {face_ID_1: np.array([vertice_ID_k, vertice_ID_l, vertice_ID_m]),
                    face_ID_2: np.array([...]),
                    ...
                    face_ID_n: np.array([...])}
        where face_ID_i is unique integer ID of face and
        vertice_ID_j is unique integer ID of vertice that belongs to particular face

        :param faces: dict
               face_ID_i: uint32
               vertice_ID_j: uint32
        """
        self._faces = {np.uint32(xx): np.array([np.uint32(yy) for yy in faces[xx]]) for xx in faces}

    @property
    def normals(self):
        """
        returns dictionary containing keywords corresponding to keywords of faces dictionary
        where keywords are the same unique integer IDs and normalised outward facing normals in numpy array
        usage: xy.normals

        :return: dict
        shape: normals = {face_ID_1: np.array([normal_x1, normal_y1, normal_z1]),
                          face_ID_2: np.array([normal_x2, normal_y2, normal_z2]),
                          ...
                          face_ID_n: np.array([normal_xn, normal_yn, normal_zn])}
        """
        return self._normals

    @normals.setter
    def normals(self, normals):
        """
        normals setter
        usage: xy.normals = new_normals
        input dictionary has to contain keywords corresponding with unique integer keywords of faces dictionary
        and numpy.array of normalized outward pointing normal vectors
        normals dictionary has to be in shape:
        normals = {face_ID_1: np.array([normal_x1, normal_y1, normal_z1]),
                   face_ID_2: np.array([normal_x2, normal_y2, normal_z2]),
                   ...
                   face_ID_n: np.array([normal_xn, normal_yn, normal_zn])}
        where face_ID_i is unique integer ID of face and
        [normal_x1, normal_y1, normal_z1] is normal vector

        :param normals: dict
               face_ID_i: uint32
               normal_xyzj: float64
        """
        self._normals = {np.uint32(xx): np.array([np.uint32(yy) for yy in normals[xx]]) for xx in normals}

    @property
    def temperatures(self):
        """
        returns temperature dictionary where keywords are unique integer keywords of faces and values are effective
        temperatures of faces
        usage: xy.temperatures

        :return:dict
        shape: {face_ID_1: t_eff1, ..., face_ID_n: t_effn}
        """
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures):
        """
        temperatures setter
        usage: xy.temperatures = new_temperatures
        input temperatures dictionary has to contain keywords corresponding with unique integer keywords of faces
        dictionary and temperatures
        :shape: {face_ID_1: t_eff1, ..., face_ID_n: t_effn}

        :param temperatures: dict
        """
        self._temperatures = {np.uint32(xx): np.float32(xx) for xx in temperatures}
        # self._temperatures = temperatures

    @property
    def mass_unit(self):
        """
        returns default mass unit
        usage: xy.mass_unit

        :return: astropy.unit.quantity.Quantity
        """
        return self._mass_unit

    @property
    def temperature_unit(self):
        """
        returns default unit of temperature
        usage: xy.temperature_unit

        :return: astropy.unit.quantity.Quantity
        """
        return self._temperature_unit

    @property
    def distance_unit(self):
        """
        returns default unit of length
        usage: xy.length_unit

        :return: astropy.unit.quantity.Quantity
        """
        return self._distance_unit

    @property
    def time_unit(self):
        """
        returns default unit of time
        usage: xy.time_unit

        :return: astropy.unit.quantity.Quantity
        """
        return self._time_unit
