import numpy as np

from elisa import utils
from elisa.base.transform import SpotParameters
from elisa.utils import is_empty


class Spot(object):
    """
    Spot data container.

    :param log_g: numpy.array

    """
    MANDATORY_KWARGS = ["longitude", "latitude", "angular_radius", "temperature_factor"]
    OPTIONAL_KWARGS = ["discretization_factor"]
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs=kwargs, kwarglist=Spot.ALL_KWARGS, instance=Spot)
        utils.check_missing_kwargs(Spot.MANDATORY_KWARGS, kwargs, instance_of=Spot)
        kwargs = self.transform_input(**kwargs)

        # supplied parameters
        self.discretization_factor = np.nan
        self.latitude = np.nan
        self.longitude = np.nan
        self.angular_radius = np.nan
        self.temperature_factor = np.nan

        # container parameters
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
        self.log_g = np.array([])

        self.init_properties(**kwargs)

    @staticmethod
    def transform_input(**kwargs):
        return SpotParameters.transform_input(**kwargs)

    def calculate_areas(self):
        """
        Returns areas of each face of the spot build_surface.
        :return: ndarray:

        ::

            numpy.array([area_1, ..., area_n])
        """
        return utils.triangle_areas(triangles=self.faces, points=self.points)

    def init_properties(self, **kwargs):
        for key in kwargs:
            set_val = kwargs.get(key)
            setattr(self, key, set_val)

    def kwargs_serializer(self):
        """
        Serializer and return mandatory kwargs of sefl (Spot) instance to dict.
        :return: Dict; { kwarg: value }
        """
        return {kwarg: getattr(self, kwarg) for kwarg in self.MANDATORY_KWARGS if not is_empty(getattr(self, kwarg))}
