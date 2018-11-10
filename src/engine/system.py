import numpy as np
import logging
import gc

from abc import ABCMeta, abstractmethod
from astropy import units as u
from engine import units as U
from engine import utils
from scipy.optimize import fsolve
from copy import copy
from scipy.spatial import KDTree

#temporary
from time import time


class System(metaclass=ABCMeta):
    """
    Abstract class defining System
    see https://docs.python.org/3.5/library/abc.html for more informations
    """

    ID = 1
    KWARGS = []
    OPTIONAL_KWARGS = []
    ALL_KWARGS = KWARGS + OPTIONAL_KWARGS

    def __init__(self, name=None, **kwargs):
        self._logger = logging.getLogger(System.__name__)

        # default params
        self._gamma = None

        if name is None:
            self._name = str(System.ID)
            self._logger.debug("Name of class instance {} set to {}".format(System.__name__, self._name))
            System.ID += 1
        else:
            self._name = str(name)

    @property
    def name(self):
        """
        name of object initialized on base of this abstract class

        :return: str
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        setter for name of system

        :param name: str
        :return:
        """
        self._name = str(name)

    @property
    def gamma(self):
        """
        system center of mass radial velocity in default velocity unit

        :return: numpy.float
        """
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        """
        system center of mass velocity
        expected type is astropy.units.quantity.Quantity, numpy.float or numpy.int othervise TypeError will be raised
        if unit is not specified, default velocity unit is assumed

        :param gamma: numpy.float/numpy.int/astropy.units.quantity.Quantity
        :return: None
        """
        if isinstance(gamma, u.quantity.Quantity):
            self._gamma = np.float64(gamma.to(U.VELOCITY_UNIT))
        elif isinstance(gamma, (int, np.int, float, np.float)):
            self._gamma = np.float64(gamma)
        else:
            raise TypeError('Value of variable `gamma` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @abstractmethod
    def compute_lc(self):
        pass

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def init(self):
        pass

    def _solver(self, fn, condition, *args, **kwargs):
        """
        will solve fn implicit function taking args by using scipy.optimize.fsolve method and return
        solution if satisfy conditional function

        # final spot containers contain their own points and simplex remaped from zero
        # final object contain only points w/o spots points and simplex (triangulation)
        # je tato poznamka vhodna tu? trosku metie

        :param fn: function
        :param condition: function
        :param args: tuple
        :return: float (np.nan), bool
        """
        # precalculation of auxiliary values
        solution, use = np.nan, False
        scipy_solver_init_value = np.array([1. / 10000.])
        try:
            solution, _, ier, mesg = fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-10)
            if ier == 1 and not np.isnan(solution[0]):
                solution = solution[0]
                use = True if 1e15 > solution > 0 else False
            else:
                self._logger.warning('Solution in implicit solver was not found, cause: {}'.format(mesg))
        except Exception as e:
            self._logger.debug("Attempt to solve function {} finished w/ exception: {}".format(fn.__name__, str(e)))
            use = False

        return (solution, use) if condition(solution, *args, **kwargs) else (np.nan, False)

    @abstractmethod
    def build_mesh(self, *args, **kwargs):
        """
        abstract method for creating surface points

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build_faces(self, *args, **kwargs):
        """
        abstract method for building body surface from given set of points in already calculated and stored in
        object.points

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build_surface(self, *args, **kwargs):
        """
        abstract method which builds surface from ground up including points and faces of surface and spots

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def build_surface_map(self, *args, **kwargs):
        """
        abstract method which calculates surface maps for surface faces of given body (eg. temperature or gravity
        acceleration map)

        :param args:
        :param kwargs:
        :return:
        """
        pass

    def _incorporate_spots_mesh(self, component_instance=None):
        if not component_instance.spots:
            return
        self._logger.info("Incorporating spot points to component {} mesh".format(component_instance.name))

        if component_instance is None:
            raise ValueError('Object instance was not given.')

        vertices_map = [{"enum": -1} for _ in component_instance.points]
        all_points = copy(component_instance.points)

        # average spacing of component surface points
        avsp = utils.average_spacing(component_instance.polar_radius, component_instance.discretization_factor)

        for spot_index, spot in component_instance.spots.items():
            # average spacing in spot points
            avsp_spot = utils.average_spacing(spot.points, spot.angular_density)
            vertices_to_remove, vertices_test = [], []

            # find nearest points to spot alt center
            tree = KDTree(all_points)
            distances, indices = tree.query(spot.boundary_center, k=len(all_points))

            max_dist_to_object_point = spot.max_size + (0.5 * avsp)
            max_dist_to_spot_point = spot.max_size + (0.1 * avsp_spot)

            # removing star points in spot
            for dist, ix in zip(distances, indices):
                if dist > max_dist_to_object_point:
                    # break, because distancies are ordered by size, so there is no more points of object that
                    # have to be removed
                    break

                if vertices_map[ix]["enum"] >= 0 and dist > max_dist_to_spot_point:
                    continue
                vertices_to_remove.append(ix)

            # simplices of target object for testing whether point lying inside or not of spot boundary, removing
            # duplicate points on the spot border
            # kedze vo vertice_map nie su body skvrny tak toto tu je zbytocne viac menej
            vertices_to_remove = list(set(vertices_to_remove))

            # points and vertices_map update
            if vertices_to_remove:
                _points, _vertices_map = list(), list()

                for ix, vertex in list(zip(range(0, len(all_points)), all_points)):
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

                all_points = copy(_points)
                vertices_map = copy(_vertices_map)

        separated_points = self.split_points_of_spots_and_component(all_points, vertices_map, component_instance)
        self.setup_component_instance_points(component_instance, separated_points)

    @classmethod
    def split_points_of_spots_and_component(cls, points, vertices_map, component_instance):
        points = np.array(points)
        component_points = {
            "object": points[np.where(np.array(vertices_map) == {'enum': -1})[0]]
        }
        # fixme: remove spots with no points left
        spots_points = {
            "{}".format(i): points[np.where(np.array(vertices_map) == {'enum': i})[0]]
            for i in range(len(component_instance.spots))
            if len(np.where(np.array(vertices_map) == {'enum': i})[0]) > 0
        }
        return {**component_points, **spots_points}

    @staticmethod
    def setup_component_instance_points(component_instance, points):
        component_instance.points = points.pop("object")
        for spot_index, spot_points in points.items():
            component_instance.spots[int(spot_index)].points = points[spot_index]

    @staticmethod
    def _return_all_points(component_instance, return_vertices_map=False):
        """
        function returns all surface point and faces optionally with corresponding map of vertices
        :param component_instance: Star object
        :return: array - all surface points including star and surface points
        """
        points = copy(component_instance.points)
        for spot_index, spot_instance in component_instance.spots.items():
            points = np.concatenate([points, spot_instance.points])

        if return_vertices_map:
            vertices_map = [{"type": "object", "enum": -1}] * len(component_instance.points)
            for spot_index, spot_instance in component_instance.spots.items():
                vertices_map = np.concatenate(
                    [vertices_map, [{"type": "spot", "enum": spot_index}] * len(spot_instance.points)]
                )
            return points, vertices_map
        return points

    @staticmethod
    def _initialize_model_container(vertices_map):
        """
        initializes basic data structure `model` objects that will contain faces divided by its origin (star or spots)
        and data structure containing spot candidates with its index, centre,
        :param vertices_map:
        :return:
        """
        model = {"object": [], "spots": {}}
        spot_candidates = {"simplex": {}, "com": {}, "3rd_enum": {}, "ix": {}}
        spots_instance_indices = list(set([vertices_map[ix]["enum"] for ix, _ in enumerate(vertices_map)
                                           if vertices_map[ix]["enum"] >= 0]))
        for spot_index in spots_instance_indices:
            model["spots"][spot_index] = []
            for key in ["com", "3rd_enum", "ix"]:
                spot_candidates[key][spot_index] = []
        return model, spot_candidates

    @staticmethod
    def _get_spots_references(vertices_map, simplex):
        reference_to_spot, trd_enum = None, None
        # variable trd_enum is enum index of 3rd corner of face;

        if vertices_map[simplex[-1]]["enum"] == vertices_map[simplex[0]]["enum"]:
            reference_to_spot = vertices_map[simplex[-1]]["enum"]
            trd_enum = vertices_map[simplex[1]]["enum"]
        elif vertices_map[simplex[0]]["enum"] == vertices_map[simplex[1]]["enum"]:
            reference_to_spot = vertices_map[simplex[0]]["enum"]
            trd_enum = vertices_map[simplex[-1]]["enum"]
        elif vertices_map[simplex[1]]["enum"] == vertices_map[simplex[-1]]["enum"]:
            reference_to_spot = vertices_map[simplex[1]]["enum"]
            trd_enum = vertices_map[simplex[0]]["enum"]
        return reference_to_spot, trd_enum

    @staticmethod
    def _resolve_spot_candidates(model, spot_candidates, component_instance, faces, components_distance=None):
        for spot_ix in spot_candidates["com"].keys():
            # get center and size of current spot candidate
            center = component_instance.spots[spot_ix].boundary_center
            size = component_instance.spots[spot_ix].max_size

            # compute distance of all center of mass of faces of current
            # spot candidate to the center of this candidate
            dists = [np.linalg.norm(np.array(com) - np.array(center)) for com in spot_candidates["com"][spot_ix]]

            # test if dist is smaller as current spot size;
            # if dist is smaller, then current face belongs to spots otherwise face belongs to t_object itself
            for idx, dist in enumerate(dists):
                simplex_ix = spot_candidates["ix"][spot_ix][idx]
                if dist < size:
                    model["spots"][spot_ix].append(np.array(faces[simplex_ix]))
                else:
                    # make the same computation for 3rd vertex of face
                    # it might be confusing, but spot candidate is spot where 2 of 3 vertex of one face belong to
                    # first spot, and the 3rd index belongs to another (neighbour) spot
                    # it has to be alos tested, whether face finally do not belongs to spot candidate;
                    trd_spot_index = spot_candidates["3rd_enum"][spot_ix][idx]
                    trd_center = component_instance.spots[trd_spot_index].boundary_center
                    trd_size = component_instance.spots[trd_spot_index].max_size
                    com = spot_candidates["com"][spot_ix][idx]
                    dist = np.linalg.norm(np.array(com) - np.array(trd_center))

                    if dist < trd_size:
                        model["spots"][trd_spot_index].append(np.array(faces[simplex_ix]))
                    else:
                        model["object"].append(np.array(faces[simplex_ix]))
        gc.collect()
        return model

    @classmethod
    def _resolve_obvious_spots(cls, points, faces, model, spot_candidates, vmap, component_instance):
        for simplex, face, ix in list(zip(faces, points[faces], range(faces.shape[0]))):
            # test if each point belongs to spot
            if {'spot'} == set([vmap[simplex[_i]]["type"] for _i in range(3)]):
                # if each point belongs to the same spot, then it is for sure face of that spot
                if vmap[simplex[0]]["enum"] == vmap[simplex[1]]["enum"] == vmap[simplex[2]]["enum"]:
                    model["spots"][vmap[simplex[0]]["enum"]].append(np.array(simplex))
                else:
                    # if at least one of points of face belongs to different spot, we have to test
                    # which one of those spots current face belongs to
                    reference_to_spot, trd_enum = cls._get_spots_references(vmap, simplex)

                    if reference_to_spot is not None:
                        spot_candidates["com"][reference_to_spot].append(np.average(face, axis=0))
                        spot_candidates["3rd_enum"][reference_to_spot].append(trd_enum)
                        spot_candidates["ix"][reference_to_spot].append(ix)
            # if at least one of point belongs to the spot and another one to the stellar body, it is neccesary
            # to compute a distance of center of mass  of such surface element from the center of spot
            # and compare whether it is within the spot or outside
            elif {'spot', 'object'}.issubset(set([vmap[simplex[_i]]["type"] for _i in range(3)])):
                simplex_indices_of_spots = [_i for _i in range(3) if vmap[simplex[_i]]["type"] == 'spot']
                spots_enum = np.array([vmap[simplex[_i]]["enum"] for _i in range(3)])[simplex_indices_of_spots]
                add_to_model = True
                for spot_ix in spots_enum:
                    center = component_instance.spots[spot_ix].boundary_center
                    size = component_instance.spots[spot_ix].max_size
                    dist = np.linalg.norm(np.array(np.average(face, axis=0)) - np.array(center))
                    if dist < size:
                        add_to_model = False
                        model["spots"][spot_ix].append(np.array(simplex))
                        break
                if add_to_model:
                    model["object"].append(np.array(simplex))
            else:
                # anything else
                model["object"].append(np.array(simplex))

        gc.collect()
        return model, spot_candidates

    @classmethod
    def _split_spots_and_component_faces(cls, points, faces, model, spot_candidates, vmap, component_instance,
                                         components_distance=None):
        """
        function that sorts faces to model data structure by distinguishing if it belongs to star or spots

        :param components_distance: float
        :param points: array (N_points * 3) - all points of surface
        :param faces: array (N_faces * 3) - all faces of the surface
        :param model: dict - data structure for faces sorting
        :param spot_candidates: initialised data structure for spot candidates
        :param vmap: vertice map
        :param component_instance: Star object
        :return:
        """
        model, spot_candidates = \
            cls._resolve_obvious_spots(points, faces, model, spot_candidates, vmap, component_instance)
        model = cls._resolve_spot_candidates(model, spot_candidates, component_instance, faces,
                                             components_distance=components_distance)
        # converting lists in model to numpy arrays
        model['object'] = np.array(model['object'])
        for spot_ix in spot_candidates["com"].keys():
            model['spots'][spot_ix] = np.array(model['spots'][spot_ix])

        return model

    def _remove_overlaped_spots(self, vertices_map, component_instance):
        # remove spots that are totaly overlaped
        spots_instance_indices = list(set([vertices_map[ix]["enum"] for ix, _ in enumerate(vertices_map)
                                           if vertices_map[ix]["enum"] >= 0]))
        for spot_index, _ in list(component_instance.spots.items()):
            if spot_index not in spots_instance_indices:
                self._logger.warning("Spot with index {} doesn't contain any face and will be removed "
                                     "from component {} spot list".format(spot_index, component_instance.name))
                component_instance.remove_spot(spot_index=spot_index)
        gc.collect()

    def _remap_surface_elements(self, model, component_instance, points_to_remap):
        """
        function remaps all surface points (`points_to_remap`) and faces (star and spots) according to the `model`

        :param model: dict - list of indices of points in `points_to_remap` divided into star and spots sublists
        :param component_instance: Star object
        :param points_to_remap: array of all surface points (star + points used in
        `BinarySystem._split_spots_and_component_faces`)
        :return:
        """
        # remapping points and faces of star
        self._logger.debug(
            "Changing value of parameter points of "
            "component {}".format(component_instance.name))
        indices = np.unique(model["object"])
        component_instance.points = points_to_remap[indices]

        self._logger.debug(
            "Changing value of parameter faces "
            "component {}".format(component_instance.name))

        points_length = np.shape(points_to_remap)[0]
        remap_list = np.empty(points_length, dtype=int)
        remap_list[indices] = np.arange(np.shape(indices)[0])
        component_instance.faces = remap_list[model["object"]]

        # remapping points and faces of spots
        for spot_index, _ in list(component_instance.spots.items()):
            self._logger.debug(
                "Changing value of parameter points of spot {} / "
                "component {}".format(spot_index, component_instance.name))
            # get points currently belong to the given spot
            indices = np.unique(model["spots"][spot_index])
            component_instance.spots[spot_index].points = points_to_remap[indices]

            self._logger.debug(
                "Changing value of parameter faces of spot {} / "
                "component {}".format(spot_index, component_instance.name))

            remap_list = np.empty(points_length, dtype=int)
            remap_list[indices] = np.arange(np.shape(indices)[0])
            component_instance.spots[spot_index].faces = remap_list[model["spots"][spot_index]]
        gc.collect()
