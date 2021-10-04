import numpy as np

from . transform import PulsationModeProperties
from .. import utils, const as c, units
from .. logger import getLogger
from .. import settings

logger = getLogger('pulse.mode')


class PulsationMode(object):
    """
    Pulsation mode data container.
    """
    MANDATORY_KWARGS = ["l", "m", "amplitude", "frequency"]

    OPTIONAL_KWARGS = ["start_phase", 'mode_axis_theta', 'mode_axis_phi', 'temperature_perturbation_phase_shift',
                       'horizontal_to_radial_amplitude_ratio', 'temperature_amplitude_factor', 'tidally_locked']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs=kwargs, kwarglist=PulsationMode.ALL_KWARGS, instance=self.__class__)
        utils.check_missing_kwargs(PulsationMode.MANDATORY_KWARGS, kwargs, instance_of=PulsationMode)
        kwargs = self.transform_input(**kwargs)

        # get logger
        logger.info(f"initialising object {self.__class__.__name__}")
        logger.debug(f"setting property components of class instance {self.__class__.__name__}")

        # self._n = np.nan
        self.l = np.nan
        self.m = np.nan
        self.amplitude = np.nan
        self.frequency = np.nan
        self.start_phase = 0
        self.mode_axis_theta = 0
        self.mode_axis_phi = 0
        self.temperature_perturbation_phase_shift = settings.DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT
        self.horizontal_to_radial_amplitude_ratio = None
        self.tidally_locked = False
        self.horizontal_to_radial_amplitude_ratio = None
        self.temperature_amplitude_factor = None
        # phase shift in radians between surface geometry
        # perturbation and temperature perturbations

        # dimensionless normalized amplitudes
        self.radial_amplitude = None  # in distance units
        self.horizontal_amplitude = None  # in distance units

        # surface related aux variables
        self.points = None  # rotated spherical coordinates aligned with pulsation axis

        self.point_harmonics = None
        self.point_harmonics_derivatives = None
        self.complex_displacement = None
        self.tilt_phi = None
        self.tilt_theta = None

        self.init_properties(**kwargs)

        self.angular_frequency = c.FULL_ARC * self.frequency

        # spherical harmonics renormalization constant to rms = 1
        # TODO: this is a constant
        self.renorm_const = 2 * c.PI ** 0.5
        self.validate_mode()

    @property
    def default_input_units(self):
        """
        Returns set of default units of intialization parameters, in case, when provided without units.

        :return: elisa.units.DefaultPulsationsInputUnits;
        """
        return units.DefaultPulsationsInputUnits

    @property
    def default_internal_units(self):
        """
        Returns set of internal units of Star parameters.

        :return: elisa.units.DefaultPulsationsUnits;
        """
        return units.DefaultPulsationsUnits

    @staticmethod
    def transform_input(**kwargs):
        """
        Transform and validate input kwargs.

        :param kwargs: Dict;
        :return: Dict;
        """
        return PulsationModeProperties.transform_input(**kwargs)

    def validate_mode(self):
        if np.abs(self.m) > self.l:
            raise ValueError(f'Absolute value of azimuthal order m: {self.m} cannot '
                             f'be higher than degree of the mode l: {self.l}.')

    def init_properties(self, **kwargs):
        """
        Setup system properties from input.

        :param kwargs: Dict; all supplied input properties
        """
        logger.debug(f"initialising properties of PulsationMode, values: {kwargs}")
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])
