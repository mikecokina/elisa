from abc import ABCMeta, abstractmethod
from astropy import units as u
import numpy as np
import logging
from engine import units as U
from engine import utils
from scipy.optimize import fsolve
from copy import copy
from scipy.spatial import KDTree

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class System(object):
    """
    Abstract class defining System
    see https://docs.python.org/3.5/library/abc.html for more informations
    """

    __metaclass__ = ABCMeta

    ID = 1
    KWARGS = []

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

    def solver(self, fn, condition, *args, **kwargs):
        """
        will solve fn implicit function taking args by using scipy.optimize.fsolve method and return
        solution if satisfy conditional function

        :param fn: function
        :param condition: function
        :param args: tuple
        :return: float (np.nan), bool
        """
        solution, use = np.nan, False
        scipy_solver_init_value = np.array([1. / 10000.])
        try:
            solution, _, ier, _ = fsolve(fn, scipy_solver_init_value, full_output=True, args=args,
                                                        xtol=1e-12)
            if ier == 1 and not np.isnan(solution[0]):
                solution = solution[0]
                use = True if 1e15 > solution > 0 else False
        except Exception as e:
            self._logger.debug("Attempt to solve function {} finished w/ exception: {}".format(fn.__name__, str(e)))
            use = False

        return (solution, use) if condition(solution, *args, **kwargs) else (np.nan, False)

    def incorporate_spots_to_surface(self, component_instance=None, build_surface_fn=None, **kwargs):
        if component_instance is None:
            raise ValueError('Object instance was not given.')
        if build_surface_fn is None:
            raise ValueError('Function for building surfaces was not specified.')

        vertices_map = [{"type": "object", "enum": -1} for _ in component_instance.points]
        points = copy(component_instance.points)

        # average spacing of component surface points
        avsp = utils.average_spacing(data=component_instance.points, neighbours=6)

        for spot_index, spot in component_instance.spots.items():
            # average spacing in spot points
            avsp_spot = utils.average_spacing(data=spot.points, neighbours=6)
            vertices_to_remove, vertices_test = [], []

            # find nearest points to spot alt center
            tree = KDTree(points)
            distances, indices = tree.query(spot.boundary_center, k=len(points))

            max_dist_to_object_point = spot.max_size + (0.25 * avsp)
            max_dist_to_spot_point = spot.max_size + (0.1 * avsp_spot)

            # removing star points in spot
            for dist, ix in zip(distances, indices):
                if dist > max_dist_to_object_point:
                    # break, because distancies are ordered by size, so there is no more points of object that
                    # have to be removed
                    break

                if vertices_map[ix]["type"] == "spot" and dist > max_dist_to_spot_point:
                    continue

                vertices_to_remove.append(ix)

            # simplices of target object for testing whether point lying inside or not of spot boundary, removing
            # duplicate points on the spot border
            # kedze vo vertice_map nie su body skvrny tak toto tu je zbytocne viac menej
            # print(len(vertices_to_remove))
            vertices_to_remove = list(set(vertices_to_remove))

            # points and vertices_map update
            if len(vertices_to_remove) != 0:
                # test if index to remove from all current points from vertices_map belongs to any of spots
                # spot_indices, star_indices = [], []
                # for item in vertices_to_remove:
                #     # that cannot occurred in firt step of loop, since there is no spot
                #     # vo vertices_map nemas body skvrny preto toto je zbytovne
                #     if vertices_map[item]["type"] == "spot":
                #         print('mylim sa, je tu take')
                #         spot_indices.append(item)
                #     else:
                #         star_indices.append(item)

                _points = []
                # _vertices_map = {}
                _vertices_map = []
                # m_ix = 0

                # for ix, vertex, norm in list(zip(range(0, len(points)), points, normals)):
                for ix, vertex in list(zip(range(0, len(points)), points)):
                    if ix in vertices_to_remove:
                        # skip point if is marked for removal
                        continue

                    # append only points of currrent object that do not intervent to spot
                    # [current, since there should be already spot from previous iteration step]
                    _points.append(vertex)
                    # _normals.append(norm)

                    # _vertices_map[m_ix] = {"type": vertices_map[ix]["type"], "enum": vertices_map[ix]["enum"]}
                    _vertices_map.append({"type": vertices_map[ix]["type"], "enum": vertices_map[ix]["enum"]})
                    # m_ix += 1

                shift = len(_points)
                # for i, vertex, norm in list(zip(range(shift, shift + len(spot.points)), spot.points, spot.normals)):
                for i, vertex in list(zip(range(shift, shift + len(spot.points)), spot.points)):
                    _points.append(vertex)
                    # _normals.append(norm)
                    # _vertices_map[i] = {"type": "spot", "enum": spot_index}
                    _vertices_map.append({"type": "spot", "enum": spot_index})

                points = copy(_points)
                vertices_map = copy(_vertices_map)
                # normals = copy(_normals)

                # del (_points, _vertices_map, _normals)
                del (_points, _vertices_map)

        points = np.array(points)
        component_instance.points = np.array(points)

        # triangulation process
        # self.build_surface(component)
        if 'component' in kwargs:
            build_surface_fn(kwargs['component'])
        else:
            build_surface_fn()

        spots_instance_indices = list(set([vertices_map[ix]["enum"]
                                           for ix, _ in enumerate(vertices_map) if vertices_map[ix]["type"] == "spot"]))

        model = {"object": [], "spots": {}}
        spot_candidates = {"simplex": {}, "com": {}, "3rd_enum": {}, "ix": {}}

        # init variables
        for spot_index in spots_instance_indices:
            model["spots"][spot_index] = []
            for key in ["com", "3rd_enum", "ix"]:
                spot_candidates[key][spot_index] = []

        # iterate over triagnulation
        for simplex, face, ix in list(zip(component_instance.faces,
                                          component_instance.points[component_instance.faces],
                                          range(component_instance.faces.shape[0]))):

            # test if each point belongs to spot
            if vertices_map[simplex[0]]["type"] == "spot" and vertices_map[simplex[1]]["type"] == "spot" \
                    and vertices_map[simplex[2]]["type"] == "spot":

                # if each point belongs to the same spot, then it is for sure face of that spot
                if vertices_map[simplex[0]]["enum"] == vertices_map[simplex[1]]["enum"] == \
                        vertices_map[simplex[2]]["enum"]:
                    model["spots"][vertices_map[simplex[0]]["enum"]].append(np.array(simplex))

                else:
                    # if at least one of points of face belongs to different spot, we have to test
                    # which one of those spots current face belongs to

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

                    if reference_to_spot is not None:
                        spot_candidates["com"][reference_to_spot].append(np.average(face, axis=0))
                        spot_candidates["3rd_enum"][reference_to_spot].append(trd_enum)
                        spot_candidates["ix"][reference_to_spot].append(ix)

            # if at least one of points belongs to star body, then it is for sure star body face
            elif vertices_map[simplex[0]]["type"] == "object" or vertices_map[simplex[1]]["type"] == "object" \
                    or vertices_map[simplex[2]]["type"] == "object":
                model["object"].append(np.array(simplex))
            else:
                model["object"].append(np.array(simplex))

        if spot_candidates["com"]:
            for spot_index in spot_candidates["com"].keys():
                # get center and size of current spot candidate
                center, size = component_instance.spots[spot_index].boundary_center, \
                               component_instance.spots[spot_index].max_size

                # compute distance of all center of mass of faces of current
                # spot candidate to the center of this candidate
                dists = [np.linalg.norm(np.array(com) - np.array(center)) for com in spot_candidates["com"][spot_index]]

                # test if dist is smaller as current spot size;
                # if dist is smaller, then current face belongs to spots otherwise face belongs to t_object itself

                for idx, dist in enumerate(dists):
                    simplex_index = spot_candidates["ix"][spot_index][idx]
                    if dist < size:
                        model["spots"][spot_index].append(np.array(component_instance.faces[simplex_index]))
                    else:
                        # make the same computation for 3rd vertex of face
                        # it might be confusing, but spot candidate is spot where 2 of 3 vertex of one face belong to
                        # first spot, and the 3rd index belongs to another (neighbour) spot
                        # it has to be alos tested, whether face finally do not belongs to spot candidate;

                        trd_spot_index = spot_candidates["3rd_enum"][spot_index][idx]

                        trd_center = component_instance.spots[trd_spot_index].boundary_center
                        trd_size = component_instance.spots[trd_spot_index].max_size

                        com = spot_candidates["com"][spot_index][idx]
                        dist = np.linalg.norm(np.array(com) - np.array(trd_center))

                        if dist < trd_size:
                            model["spots"][trd_spot_index].append(np.array(component_instance.faces[simplex_index]))
                        else:
                            model["object"].append(np.array(component_instance.faces[simplex_index]))

        # remove spots that are totaly overlaped
        for spot_index, _ in list(component_instance.spots.items()):
            if spot_index not in spots_instance_indices:
                self._logger.warning("Spot with index {} doesn't contain any face and will be removed "
                                     "from component {} spot list".format(spot_index, component_instance.name))
                component_instance.remove_spot(spot_index=spot_index)
            else:
                self._logger.debug(
                    "Changing value of parameter points of spot {} / "
                    "component {}".format(spot_index, component_instance.name))
                # get points currently belong to the given spot
                indices = list(set(np.array(model["spots"][spot_index]).flatten()))
                component_instance.spots[spot_index].points = np.array(component_instance.points[indices])

                self._logger.debug(
                    "Changing value of parameter faces of spot {} / "
                    "component {}".format(spot_index, component_instance.name))
                component_instance.spots[spot_index].faces = model["spots"][spot_index]
                remap_dict = {idx[1]: idx[0] for idx in enumerate(indices)}
                component_instance.spots[spot_index].faces = \
                    np.array(utils.remap(component_instance.spots[spot_index].faces, remap_dict))

        self._logger.debug("Changing value of parameter points of object {}".format(component_instance.name))
        indices = list(set(np.array(model["object"]).flatten()))
        component_instance.points = component_instance.points[indices]

        self._logger.debug("Changing value of parameter faces of object {}".format(component_instance.name))
        remap_dict = {idx[1]: idx[0] for idx in enumerate(indices)}

        component_instance.faces = np.array(utils.remap(model["object"], remap_dict))
