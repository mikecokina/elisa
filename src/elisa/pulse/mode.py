import numpy as np
from elisa import utils, const as c
from elisa.logger import getLogger
from elisa.pulse.transform import PulsationModeProperties
from elisa.conf import config

logger = getLogger('pulse.mode')


class PulsationMode(object):
    """
    Pulsation mode data container.
    """
    MANDATORY_KWARGS = ["l", "m", "amplitude", "frequency"]

    OPTIONAL_KWARGS = ["start_phase", 'mode_axis_theta', 'mode_axis_phi', 'temperature_perturbation_phase_shift']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs=kwargs, kwarglist=PulsationMode.ALL_KWARGS, instance=self.__class__.__name__)
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
        self.temperature_perturbation_phase_shift = config.DEFAULT_TEMPERATURE_PERTURBATION_PHASE_SHIFT
        # phase shift in radians between surface geometry
        # perturbation and temperature perturbations

        # here the time-independent, renormalized associated Legendree polynomial is stored
        self.rals = None

        self.radial_relative_amplitude = None
        self.horizontal_relative_amplitude = None

        self.init_properties(**kwargs)

        self.angular_frequency = c.FULL_ARC * self.frequency
        self.renorm_const = utils.spherical_harmonics_renormalization_constant(self.l, self.m)

        self.validate_mode()

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
