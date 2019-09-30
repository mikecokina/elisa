import gc
import numpy as np

from abc import ABCMeta
from copy import copy
from astropy import units as u

from elisa import logger, utils, units
from elisa.utils import is_empty
from elisa.base.spot import Spot


class Body(metaclass=ABCMeta):
    """
    Abstract class that defines bodies that can be modelled by this software.
    Units are imported from astropy.units module::

        see documentation http://docs.astropy.org/en/stable/units/
    """

    ID = 1

    def __init__(self, name, logger_name=None, suppress_logger=False, **kwargs):
        """
        Parameters of abstract class Body.
        """
        # initial kwargs
        self.kwargs = copy(kwargs)
        self._suppress_logger = suppress_logger
        self._logger = logger.getLogger(logger_name or self.__class__.__name__, suppress=self._suppress_logger)

        if is_empty(name):
            self._name = str(Body.ID)
            self._logger.debug(f"name of class instance {self.__class__.__name__} set to {self._name}")
            Body.ID += 1
        else:
            self._name = str(name)

        # initializing parmas to default values
        self._mass = np.nan
        self._t_eff = np.nan
        self._points = np.array([])
        self._faces = np.array([])
        self._normals = np.array([])
        self._temperatures = np.array([])
        self._synchronicity = np.nan
        self._albedo = np.nan
        self._polar_radius = np.nan
        self._areas = np.array([])
        self._discretization_factor = np.radians(3)
        self._face_centres = np.array([])
        self._spots = dict()
        self._point_symmetry_vector = np.array([])
        self.inverse_point_symmetry_matrix = np.array([])
        self.base_symmetry_points_number = 0
        self._face_symmetry_vector = np.array([])
        self.base_symmetry_faces_number = 0
        # those are used only if case of spots are used
        self.base_symmetry_points = np.array([])
        self.base_symmetry_faces = np.array([])

    def has_spots(self):
        """
        Find whether object has defined spots.

        :return: bool
        """
        return len(self._spots) > 0

    @property
    def name(self):
        """
        *<instance>* name getter
        usage: *<instance>*.name

        :return: str
        """
        return str(self._name)

    @name.setter
    def name(self, name):
        """
        :param name: str
        :return:
        """
        self._name = str(name)

    @property
    def mass(self):
        """
        :return: float
        """
        return self._mass

    @mass.setter
    def mass(self, mass):
        """
        If mass is int, np.int, float, np.float, program assumes solar mass as it's unit.
        If mass astropy.unit.quantity.Quantity instance, program converts it to default units and stores it's value in
        attribute _mass.

        :param mass: int, numpy.int, float, numpy.float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(mass, u.quantity.Quantity):
            self._mass = np.float64(mass.to(units.MASS_UNIT))
        elif isinstance(mass, (int, np.int, float, np.float)):
            self._mass = np.float64(mass * u.solMass.to(units.MASS_UNIT))
        else:
            raise TypeError('User input is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def t_eff(self):
        """
        :return: float
        """
        return self._t_eff

    @t_eff.setter
    def t_eff(self, t_eff):
        """
        This function accepts value in Any temperature unit.
        If your input is without unit, function assumes that supplied value is in Kelvins.

        :param t_eff: int, numpy.int, float, numpy.float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(t_eff, u.quantity.Quantity):
            self._t_eff = np.float64(t_eff.to(units.TEMPERATURE_UNIT))
        elif isinstance(t_eff, (int, np.int, float, np.float)):
            self._t_eff = np.float64(t_eff)
        else:
            raise TypeError('Value of `t_eff` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def points(self):
        """
        Returns dictionary of points that forms surface of Body.

        :return: numpy.array
        """
        return self._points

    @points.setter
    def points(self, points):
        """
        Setting numpy array of points that form surface of Body.
        Input dictionary has to be in shape::

            points = numpy.array([[x1 y1 z1],
                                  [x2 y2 z2],
                                   ...
                                  [xN yN zN]])

        where xi, yi, zi are cartesian coordinates of vertice i.

        :param points: numpy.array; [xi, yi, zi]: float
        :return:
        """
        self._points = np.array(points)

    @property
    def faces(self):
        """
        Returns dictionary of triangles that will create surface of body.
        Triangles are stored as list of indices of points.

        :return: numpy.array

        ::

            numpy.array([[vertice_index_k, vertice_index_l, vertice_index_m]),
            [...]),
             ...
            [...]])
        """
        return self._faces

    @faces.setter
    def faces(self, faces):
        """
        Faces dictionary has to be in shape::

            points = np.array([vertice_index_k, vertice_index_l, vertice_index_m],
                              [...],
                               ...
                              [...]]

        :param faces: numpy.array
        :return:
        """
        self._faces = faces

    @property
    def normals(self):
        """
        Returns array containing normalised outward facing normals of corresponding faces with same index.

        :return: numpy.array

        ::

            normals = numpy_array([[normal_x1, normal_y1, normal_z1],
                                   [normal_x2, normal_y2, normal_z2],
                                    ...
                                   [normal_xn, normal_yn, normal_zn]]
        """
        return self._normals

    @normals.setter
    def normals(self, normals):
        """
        Setter for normalised outward facing normals of corresponding faces with same index.
        Expected shape of normals matrix::

            normals = numpy_array([[normal_x1, normal_y1, normal_z1],
                                   [normal_x2, normal_y2, normal_z2],
                                           ...
                                   [normal_xn, normal_yn, normal_zn]]

        :param normals: dnarray
        :return:
        """
        self._normals = normals

    @property
    def areas(self):
        """
        Returns array of areas of corresponding faces.

        :return: numpy.array

        ::

            numpy.array([area_1, ..., area_n])
        """
        return self._areas

    @areas.setter
    def areas(self, areas):
        """
        Returns array of areas of corresponding faces.

        :param areas

        ::

            numpy.array([area_1, ..., area_n])

        :return:
        """
        self._areas = areas

    @property
    def temperatures(self):
        """
        Returns array of temeratures of corresponding faces.

        :return: numpy.array

        ::

            numpy.array([t_eff1, ..., t_effn])
        """
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures):
        """
        Aetter for array of temeratures of corresponding faces in shape::

            numpy.array([t_eff1, ..., t_effn])

        :param temperatures: numpy.array
        :return:
        """
        self._temperatures = temperatures

    @property
    def synchronicity(self):
        """
        Returns synchronicity parameter F = omega_rot/omega_orb.

        :return: float
        """
        return self._synchronicity

    @synchronicity.setter
    def synchronicity(self, synchronicity):
        """
        Object synchronicity (F = omega_rot/omega_orb) setter.
        Expects number input convertible to numpy float64 / float.

        :param synchronicity: float
        """
        self._synchronicity = np.float64(synchronicity)

    @property
    def albedo(self):
        """
        Returns bolometric albedo of an object (reradiated energy/ irradiance energy).

        :return: float
        """
        return self._albedo

    @albedo.setter
    def albedo(self, albedo):
        """
        Setter for bolometric albedo (reradiated energy/ irradiance energy).
        Accepts value of albedo in range (0, 1).

        :param albedo: float
        """
        if 0. <= albedo <= 1.:
            self._albedo = np.float64(albedo)
        else:
            raise ValueError(f'Parameter albedo = {albedo} is out of range (0, 1)')

    @property
    def polar_radius(self):
        """
        Returns value polar radius of an object in default unit.

        :return: float
        """
        return self._polar_radius

    @polar_radius.setter
    def polar_radius(self, polar_radius):
        """
        Expected type is astropy.units.quantity.Quantity, numpy.float or numpy.int othervise TypeError will be raised.
        If quantity is not specified, default distance unit is assumed.

        :param polar_radius: float
        :return:
        """
        if isinstance(polar_radius, u.quantity.Quantity):
            self._polar_radius = np.float64(polar_radius.to(units.DISTANCE_UNIT))
        elif isinstance(polar_radius, (int, np.int, float, np.float)):
            self._polar_radius = np.float64(polar_radius)
        else:
            raise TypeError('Value of variable `polar radius` is not (numpy.)int or (numpy.)float '
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
        :param discretization_factor: float
        :return:
        """
        if isinstance(discretization_factor, u.quantity.Quantity):
            self._discretization_factor = np.float64(discretization_factor.to(units.ARC_UNIT))
        elif isinstance(discretization_factor, (int, np.int, float, np.float)):
            self._discretization_factor = np.radians(np.float64(discretization_factor))
        else:
            raise TypeError('Input of variable `discretization_factor` is not (numpy.)int'
                            ' or (numpy.)float nor astropy.unit.quantity.Quantity instance.')

    @property
    def face_centres(self):
        """
        Returns coordinates of centres of corresponding faces.

        :return: numpy.array

        ::

            numpy.array([[face_centrex1, face_centrey1, face_centrez1],
                           [face_centrex2, face_centrey2, face_centrez2],
                            ...,
                           [face_centrexN, face_centrey2, face_centrez2]])
        """
        return self._face_centres

    @face_centres.setter
    def face_centres(self, centres):
        """
        :param centres: numpy.array

        ::

                numpy.array([[face_centrex1, face_centrey1, face_centrez1],
                             [face_centrex2, face_centrey2, face_centrez2],
                              ...,
                             [face_centrexN, face_centrey2, face_centrez2]])
        :return:
        """
        if np.shape(centres)[0] != np.shape(self.faces)[0]:
            raise ValueError('Number of surface centres doesn`t equal to number of faces')
        self._face_centres = centres

    @property
    def spots(self):
        """
        :return: Dict[int, Spot]
        """
        return self._spots

    @spots.setter
    def spots(self, spots):
        """
        example of defined spots

        ::

            [
                 {"longitude": 90,
                  "latitude": 58,
                  "angular_diameter": 15,
                  "temperature_factor": 0.9},
                 {"longitude": 85,
                  "latitude": 80,
                  "angular_diameter": 30,
                  "temperature_factor": 1.05},
                 {"longitude": 45,
                  "latitude": 90,
                  "angular_diameter": 30,
                  "temperature_factor": 0.95},
             ]

        :param spots: Iterable[Dict]; definition of spots for given object
        :return:
        """
        self._spots = {idx: Spot(**spot_meta) for idx, spot_meta in enumerate(spots)} if not is_empty(spots) else dict()

    @property
    def point_symmetry_vector(self):
        """
        Vector of indices with the same length as body`s points.
        N-th value of point_symmetry_matrix indicates position of base symmetry point for given n-th point.

        :return: numpy.array
        """
        return self._point_symmetry_vector

    @point_symmetry_vector.setter
    def point_symmetry_vector(self, symmetry_vector):
        """
        Setter for vector of indices with the same length as body`s points, n-th value of point_symmetry_matrix
        indicates position of base symmetry point for given n-th point.

        :param symmetry_vector: numpy.array

        ::

            numpy.array([index_of_symmetry_point_for_point1, ..., index_of_symmetry_point_for_pointN])

        :return:
        """
        if np.shape(self.points)[0] != np.shape(symmetry_vector)[0] and not self.has_spots():
            raise ValueError(f'Length of symmetry vector {np.shape(symmetry_vector)[0]} is not '
                             f'the same as number of surface points {np.shape(self.points)[0]}')
        self._point_symmetry_vector = symmetry_vector

    @property
    def face_symmetry_vector(self):
        """
        Vector of indices with the same length as body`s faces, n-th value of face_symmetry_matrix indicates position
        of base symmetry face for given n-th point.

        :return: numpy.array
        """
        return self._face_symmetry_vector

    @face_symmetry_vector.setter
    def face_symmetry_vector(self, symmetry_vector):
        """
        Setter for vector of indices with the same length as body`s faces, n-th value of face_symmetry_matrix
        indicates position of base symmetry face for given n-th point.

        :param symmetry_vector: numpy.array;

        ::

            numpy.array([index_of_symmetry_face1, ..., index_of_symmetry_faceN])

        :return:
        """
        if np.shape(self.faces)[0] != np.shape(symmetry_vector)[0]:
            raise ValueError(f'Length of symmetry vector {np.shape(symmetry_vector)[0]} is not '
                             f'the same as number of surface faces {np.shape(self.faces)[0]}')
        self._face_symmetry_vector = symmetry_vector

    # <units> **********************************************************************************************************
    @property
    def mass_unit(self):
        """
        Returns default mass unit.

        :return: astropy.unit.quantity.Quantity
        """
        return units.MASS_UNIT

    @property
    def temperature_unit(self):
        """
        Returns default unit of temperature.

        :return: astropy.unit.quantity.Quantity
        """
        return units.TEMPERATURE_UNIT

    @property
    def distance_unit(self):
        """
        Rreturns default unit of length.

        :return: astropy.unit.quantity.Quantity
        """
        return units.DISTANCE_UNIT

    @property
    def time_unit(self):
        """
        Returns default unit of time.

        :return: astropy.unit.quantity.Quantity
        """
        return units.TIME_UNIT

    @property
    def arc_unit(self):
        """
        Returns default unit of time.

        :return: astropy.unit.quantity.Quantity
        """
        return units.ARC_UNIT
    # </units> *********************************************************************************************************

    def remove_spot(self, spot_index: int):
        """
        Remove n-th spot index of object.

        :param spot_index: int
        :return:
        """
        del (self._spots[spot_index])

    def calculate_normals(self, points, faces, centres=None, com=None):
        """
        Returns outward facing normal unit vector for each face of stellar surface.

        :return: numpy.array:

        ::

            numpy.array([[normal_x1, normal_y1, normal_z1],
                         [normal_x2, normal_y2, normal_z2],
                          ...
                         [normal_xn, normal_yn, normal_zn]])
        """
        normals = np.array([np.cross(points[xx[1]] - points[xx[0]], points[xx[2]] - points[xx[0]]) for xx in faces])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        cntr = self.calculate_surface_centres(points, faces) if is_empty(centres) else copy(centres)

        corr_centres = cntr - np.array([com, 0, 0])[None, :]

        # making sure that normals are properly oriented near the axial planes
        sgn = np.sign(np.sum(np.multiply(normals, corr_centres), axis=1))

        return normals * sgn[:, None]

    def set_all_normals(self, com=None):
        """
        Function calculates normals for each face of given body (including spots) and assign it to object.

        :param com: numpy.array
        :return:
        """
        self.normals = self.calculate_normals(points=self.points, faces=self.faces, centres=self.face_centres, com=com)
        if self.has_spots():
            for spot_index in self.spots:
                self.spots[spot_index].normals = self.calculate_normals(points=self.spots[spot_index].points,
                                                                        faces=self.spots[spot_index].faces,
                                                                        centres=self.spots[spot_index].face_centres,
                                                                        com=com)

    def set_all_surface_centres(self):
        """
        Calculates all surface centres for given body(including spots) and assign to object as `face_centers` property

        :return:
        """
        self.face_centres = self.calculate_surface_centres()
        if self.has_spots():
            for spot_index, spot in self.spots.items():
                spot.face_centres = \
                    self.calculate_surface_centres(points=spot.points,
                                                   faces=spot.faces)

    def calculate_surface_centres(self, points=None, faces=None):
        """
        Returns centers of every surface face.
        If `points` is not supplied, parameter of self instance is used for both, `points` and `faces`.

        :return: numpy.array:

        ::

            numpy.array([[center_x1, center_y1, center_z1],
                         [center_x2, center_y2, center_z2],
                          ...
                         [center_xn, center_yn, center_zn]])
        """
        if is_empty(points):
            points = self.points
            faces = self.faces
        return np.average(points[faces], axis=1)

    def calculate_areas(self):
        """
        Returns areas of each face of the star surface.

        :return: numpy.array:

        ::

            numpy.array([area_1, ..., area_n])
        """
        if len(self.faces) == 0 or len(self.points) == 0:
            raise ValueError('Faces or/and points of object {self.name} have not been set yet.\n'
                             'Run build method first.')
        # FIXME: why the hell is this temporary?????
        if not self.has_spots():  # temporary
            base_areas = utils.triangle_areas(self.faces[:self.base_symmetry_faces_number],
                                              self.points[:self.base_symmetry_points_number])
            return base_areas[self.face_symmetry_vector]

        else:
            return utils.triangle_areas(self.faces, self.points)

    def calculate_all_areas(self):
        """
        Calculates areas for all faces on the surface including spots and assigns values to its corresponding variables.

        :return:
        """
        self.areas = self.calculate_areas()
        if self.has_spots():
            for spot_index, spot in self.spots.items():
                spot.areas = spot.calculate_areas()

    def return_whole_surface(self):
        """
        Returns all points and faces of the whole star.

        :return: Tuple[numpy.array, numpy.array]
        """
        ret_points = copy(self.points)
        ret_faces = copy(self.faces)
        if self.has_spots():
            for spot_index, spot in self.spots.items():
                n_points = np.shape(ret_points)[0]
                ret_points = np.append(ret_points, spot.points, axis=0)
                ret_faces = np.append(ret_faces, spot.faces + n_points, axis=0)

        return ret_points, ret_faces

    def return_all_points(self, return_vertices_map=False):
        """
        Function returns all surface point and faces optionally with corresponding map of vertices.

        :param return_vertices_map: bool
        :param self: Star instance
        :return: Tuple[numpy.array, Any]: [all surface points including star and surface points, vertices map or None]
        """
        points = copy(self.points)
        for spot_index, spot_instance in self.spots.items():
            points = np.concatenate([points, spot_instance.points])

        if return_vertices_map:
            vertices_map = [{"type": "object", "enum": -1}] * len(self.points)
            for spot_index, spot_instance in self.spots.items():
                vertices_map = np.concatenate(
                    [vertices_map, [{"type": "spot", "enum": spot_index}] * len(spot_instance.points)]
                )
            return points, vertices_map
        return points, None

    def setup_body_points(self, points):
        """
        Setup points for Star instance and spots based on input `points` Dict object.
        Such `points` map looks like following

        ::

            {
                "object": [[point0_x, point0_y, point0_z], ..., [pointN_x, pointN_y, pointN_z]],
                "0": [<points>],
                "1": [<points>]...
            },

            where `object` contain numpy.array of object points and indices points for given spot.

        :param points: Dict[str, numpy.array]
        :return:
        """
        self.points = points.pop("object")
        for spot_index, spot_points in points.items():
            self.spots[int(spot_index)].points = points[spot_index]

    def remove_overlaped_spots_by_vertex_map(self, vertices_map):
        """
        Remove spots of Start object that are totally overlapped by another spot.

        :param vertices_map: List or numpy.array
        :return:
        """
        # remove spots that are totaly overlaped
        spots_instance_indices = list(set([vertices_map[ix]["enum"] for ix, _ in enumerate(vertices_map)
                                           if vertices_map[ix]["enum"] >= 0]))
        for spot_index, _ in list(self.spots.items()):
            if spot_index not in spots_instance_indices:
                self._logger.warning(f"spot with index {spot_index} doesn't contain Any face "
                                     f"and will be removed from component {self.name} spot list")
                self.remove_spot(spot_index=spot_index)
        gc.collect()

    def remap_surface_elements(self, model, points_to_remap):
        """
        Function remaps all surface points (`points_to_remap`) and faces (star and spots) according to the `model`.

        :param model: dict - list of indices of points in `points_to_remap` divided into star and spots sublists
        :param points_to_remap: array of all surface points (star + points used in `_split_spots_and_component_faces`)
        :return:
        """
        # remapping points and faces of star
        self._logger.debug(f"changing value of parameter points of component {self.name}")
        indices = np.unique(model["object"])
        self.points = points_to_remap[indices]

        self._logger.debug(f"changing value of parameter faces of component {self.name}")

        points_length = np.shape(points_to_remap)[0]
        remap_list = np.empty(points_length, dtype=int)
        remap_list[indices] = np.arange(np.shape(indices)[0])
        self.faces = remap_list[model["object"]]

        # remapping points and faces of spots
        for spot_index, _ in list(self.spots.items()):
            self._logger.debug(f"changing value of parameter points of spot {spot_index} / component {self.name}")
            # get points currently belong to the given spot
            indices = np.unique(model["spots"][spot_index])
            self.spots[spot_index].points = points_to_remap[indices]

            self._logger.debug(f"changing value of parameter faces of spot {spot_index} / component {self.name}")

            remap_list = np.empty(points_length, dtype=int)
            remap_list[indices] = np.arange(np.shape(indices)[0])
            self.spots[spot_index].faces = remap_list[model["spots"][spot_index]]
        gc.collect()

    def setup_spot_instance_discretization_factor(self, spot_instance, spot_index):
        """
        Setup discretization factor for given spot instance based on defined rules::

            - used Star discretization factor if not specified in spot
            - if spot_instance.discretization_factor > 0.5 * spot_instance.angular_diameter then factor is set to
              0.5 * spot_instance.angular_diameter

        :param spot_instance: Spot
        :param spot_index: int; spot index (has no affect on process, used for logging)
        :return:
        """
        # component_instance = getattr(self, component)
        if is_empty(spot_instance.discretization_factor):
            self._logger.debug(f'angular density of the spot {spot_index} on {self.name} component was not supplied '
                               f'and discretization factor of star {self.discretization_factor} was used.')
            spot_instance.discretization_factor = 0.9 * self.discretization_factor * units.ARC_UNIT
        if spot_instance.discretization_factor > spot_instance.angular_radius:
            self._logger.debug(f'angular density {self.discretization_factor} of the spot {spot_index} on {self.name} '
                               f'component was larger than its angular radius. Therefore value of angular density was '
                               f'set to be equal to 0.5 * angular diameter')
            spot_instance.discretization_factor = spot_instance.angular_radius * units.ARC_UNIT

    def incorporate_spots_mesh(self, component_com=None):
        """
        Based on spots definitions, evaluate spot points on Star surface and remove those points of Star itself
        which are inside of any spots. Do the same operation with spot to each other.
        Evaluation is running from index 0, what means that spot with lower index
        is overwriten by spot with higher index.

        All points are assigned to given object (Star points to Star object and Spot points to related Spot object).

        # todo: change structure flat array like [-1, -1, -1, ..., 0, ..., 1, ...]
        Defines variable `vertices_map` used in others method. Strutcure of this variable is following::

            [{"enum": -1}, {"enum": 0}, {"enum": 1}, ..., {"enum": N}].

        Enum index -1 means that points in joined array of points on current
        index position of vertices_map belongs to Star point.
        Enum indices >= 0 means the same, but for Spot.

        :param component_com: center of mass of component
        :return:
        """
        if not self.spots:
            self._logger.debug(f"not spots found, skipping incorporating spots to mesh on component {self.name}")
            return
        self._logger.debug(f"incorporating spot points to component {self.name} mesh")

        if is_empty(component_com):
            raise ValueError('Object centre of mass was not supplied')

        vertices_map = [{"enum": -1} for _ in self.points]
        # `all_component_points` do not contain points of Any spot
        all_component_points = copy(self.points)
        neck = np.min(all_component_points[:self.base_symmetry_points_number, 0])

        for spot_index, spot in self.spots.items():
            # average spacing in spot points
            vertices_to_remove, vertices_test = [], []
            cos_max_angle_point = np.cos(spot.angular_radius + 0.30 * spot.discretization_factor)
            spot_center = spot.center - np.array([component_com, 0., 0.])

            # removing star points in spot
            # all component points means just points of component NOT merged points + spots
            for ix, pt in enumerate(all_component_points):
                surface_point = all_component_points[ix] - np.array([component_com, 0., 0.])
                cos_angle = \
                    np.inner(spot_center, surface_point) / (
                        np.linalg.norm(spot_center) * np.linalg.norm(surface_point))
                # skip all points of object outside of spot
                if cos_angle < cos_max_angle_point or np.linalg.norm(pt[0] - neck) < 1e-9:
                    continue
                # mark component point (NOT point of spot) for removal if is within the spot
                vertices_to_remove.append(ix)

            # simplices of target object for testing whether point lying inside or not of spot boundary, removing
            # duplicate points on the spot border
            # kedze vo vertice_map nie su body skvrny tak toto tu je zbytocne viac menej
            vertices_to_remove = list(set(vertices_to_remove))

            # points and vertices_map update
            _points, _vertices_map = list(), list()

            for ix, vertex in list(zip(range(0, len(all_component_points)), all_component_points)):
                if ix in vertices_to_remove:
                    # skip point if is marked for removal
                    continue

                # append only points of currrent object that do not intervent to spot
                # [current, since there should be already spot from previous iteration step]
                _points.append(vertex)
                _vertices_map.append({"enum": vertices_map[ix]["enum"]})

            for vertex in spot.points:
                _points.append(vertex)
                _vertices_map.append({"enum": spot_index})

            all_component_points = copy(_points)
            vertices_map = copy(_vertices_map)

        separated_points = self.split_points_of_spots_and_component(all_component_points, vertices_map)
        self.setup_body_points(separated_points)

    def split_points_of_spots_and_component(self, points, vertices_map):
        """
        Based on vertices map, separates points to points which belong to Star object
        and points which belong to each defined Spot object.
        During the process remove overlapped spots.

        :param points: numpy.array; all points of object (spot points and component points together)
        :param vertices_map: List or numpy.array; map which define refrences of index in
                             given Iterable to object (Spot or Star).
        :return: Dict;

        ::

            {
                "object": numpy.array([[pointN_x, pointN_y, pointN_z], ...]),
                "spot_index": numpy.array([[pointM_x, pointM_y, pointM_z], ...]), ...
            }
        """
        points = np.array(points)
        component_points = {
            "object": points[np.where(np.array(vertices_map) == {'enum': -1})[0]]
        }
        self.remove_overlaped_spots_by_spot_index(
            spot_indices=set([int(val["enum"]) for val in vertices_map if val["enum"] > -1])
        )
        spots_points = {
            f"{i}": points[np.where(np.array(vertices_map) == {'enum': i})[0]] for i in range(len(self.spots))
            if len(np.where(np.array(vertices_map) == {'enum': i})[0]) > 0
        }
        return {**component_points, **spots_points}

    def remove_overlaped_spots_by_spot_index(self, spot_indices):
        """
        Remove definition and instance of those spots that are overlaped
        by another one and basically has no face to work with.

        :param spot_indices: List[int]; list of spot indices to remove
        :return:
        """
        all_spot_indices = set([int(val) for val in self.spots.keys()])
        spot_indices_to_remove = all_spot_indices.difference(spot_indices)

        for spot_index in spot_indices_to_remove:
            self.remove_spot(spot_index)

    def split_spots_and_component_faces(self, points, faces, model, spot_candidates, vmap, component_com):
        """
        Function that sorts faces to model data structure by distinguishing if it belongs to star or spots.

        :param component_com: float; center of mass of component
        :param points: numpy.array; (N_points * 3) - all points of surface
        :param faces: numpy.array; (N_faces * 3) - all faces of the surface
        :param model: Dict; data structure for faces sorting (more in docstring of method `initialize_model_container`)
        :param spot_candidates: Dict; initialised data structure for spot candidates
        :param vmap: vertice map, for more info, see docstring for `incorporate_spots_mesh` method
        :return: Dict; same as param `model`
        """
        model, spot_candidates = \
            self._resolve_obvious_spots(points, faces, model, spot_candidates, vmap)
        model = self._resolve_spot_candidates(model, spot_candidates, faces, component_com=component_com)
        # converting lists in model to numpy arrays
        model['object'] = np.array(model['object'])
        for spot_ix in self.spots:
            model['spots'][spot_ix] = np.array(model['spots'][spot_ix])
        return model

    @classmethod
    def _resolve_obvious_spots(cls, points, faces, model, spot_candidates, vmap):
        """
        Resolve those Spots/Star faces, where all tree vertices belongs to given object.
        If there are mixed vertices of any face, append it to spot candidate List.

        :param points: numpy.array; array of all points
        :param faces: numpy.array; array of all faces
        :param model: Dict; dictionary which describe object with spots as one entity
        :param spot_candidates:
        :param vmap: vertices map; for more info, see docstring for `incorporate_spots_mesh` method
        :return: Tuple[Dict, Dict]
        """

        for simplex, face_points, ix in list(zip(faces, points[faces], range(faces.shape[0]))):
            # if each point belongs to the same spot, then it is for sure face of that spot
            condition1 = vmap[simplex[0]]["enum"] == vmap[simplex[1]]["enum"] == vmap[simplex[2]]["enum"]
            if condition1:
                if 'spot' == vmap[simplex[0]]["type"]:
                    model["spots"][vmap[simplex[0]]["enum"]].append(np.array(simplex))
                else:
                    model["object"].append(np.array(simplex))
            else:
                spot_candidates["com"].append(np.average(face_points, axis=0))
                spot_candidates["ix"].append(ix)

        gc.collect()
        return model, spot_candidates

    @staticmethod
    def initialize_model_container(vertices_map):
        """
        Initializes basic data structure `model` objects that will contain faces divided by its origin (star or spots)
        and data structure containing spot candidates with its index and center point.

        Structure is based on input `verties_map`.

        Example of return Tuple

        ::

            (<class 'dict'>: {'object': [], 'spots': {0: []}}, <class 'dict'>: {'com': [], 'ix': []})

        :param vertices_map: List or ndarray; map which define refrences of index in
                             given Iterable to object (spot or Star).
                             For more info, see docstring for `incorporate_spots_mesh` method.
        :return: Tuple[Dict, Dict]
        """
        model = {"object": list(), "spots": dict()}
        spot_candidates = {"com": list(), "ix": list()}

        spots_instance_indices = list(set([vertices_map[ix]["enum"] for ix, _ in enumerate(vertices_map)
                                           if vertices_map[ix]["enum"] >= 0]))
        for spot_index in spots_instance_indices:
            model["spots"][spot_index] = list()
        return model, spot_candidates

    def _resolve_spot_candidates(
            self, model, spot_candidates, faces, component_com):
        """
        Resolves spot face candidates by comparing angular distances of face cantres and spot centres.
        In case of multiple layered spots, face is assigned to the top layer.

        :param model: Dict; initialised dictionary with placeholders which will describe object with spots as one entity
        :param spot_candidates: Dict; contain indices and center of mass of each
                                      face that is candodate to be a face of spot
        :param faces: ndarray; array of indices which defines faces from points;
        :param component_com: float; center of mass of given component
        :return: Dict; filled model from input of following structure


        ::

            {
                "object": ndarray[[point_idx_i, point_idx_j, point_idx_k], ...],
                "spots": {
                    spot_index: ndarray[[point_idx_u, point_idx_v, point_idx_w], ...]
                }
            }
        """
        # checking each candidate one at a time trough all spots
        com = np.array(spot_candidates["com"]) - np.array([component_com, 0.0, 0.0])
        cos_max_angle = {idx: np.cos(spot.angular_radius) for idx, spot in self.spots.items()}
        center = {idx: spot.center - np.array([component_com, 0.0, 0.0])
                  for idx, spot in self.spots.items()}
        for idx, _ in enumerate(spot_candidates["com"]):
            spot_idx_to_assign = -1
            simplex_ix = spot_candidates["ix"][idx]
            for spot_ix in self.spots:
                cos_angle_com = np.inner(center[spot_ix], com[idx]) / \
                                (np.linalg.norm(center[spot_ix]) * np.linalg.norm(com[idx]))
                if cos_angle_com > cos_max_angle[spot_ix]:
                    spot_idx_to_assign = spot_ix

            if spot_idx_to_assign == -1:
                model["object"].append(np.array(faces[simplex_ix]))
            else:
                model["spots"][spot_idx_to_assign].append(np.array(faces[simplex_ix]))

        gc.collect()
        return model
