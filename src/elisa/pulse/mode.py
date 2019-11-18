import numpy as np
from elisa import utils, const as c, units
from elisa.logger import getLogger
from elisa.pulse.transform import PulsationModeProperties

logger = getLogger('pulse.mode')


class PulsationMode(object):
    """
    pulsation mode data container
    """
    MANDATORY_KWARGS = ["l", "m", "amplitude", "frequency"]

    OPTIONAL_KWARGS = ["start_phase", 'mode_axis_theta', 'mode_axis_phi']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs=kwargs, kwarglist=PulsationMode.ALL_KWARGS, instance=self.__class__.__name__)
        utils.check_missing_kwargs(PulsationMode.MANDATORY_KWARGS, kwargs, instance_of=PulsationMode)
        kwargs = self.transform_input(**kwargs)

        # get logger
        logger.info(f"initialising object {self.__class__.__name__}")
        logger.debug(f"setting property components of class instance {self.__class__.__name__}")

        # # self._n = np.nan
        self.l = np.nan
        self.m = np.nan
        self.amplitude = np.nan
        self.frequency = np.nan
        self.start_phase = 0
        self.mode_axis_theta = 0
        self.mode_axis_phi = 0

        # here the time-independent, renormalized associated Legendree polynomial is stored
        self.rals = None
        self.rals_constant = None

        self.init_properties(**kwargs)

        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        for kwarg in kwargs:
            logger.debug(f"setting property {kwarg} "
                               f"of class instance {self.__class__.__name__} to {kwargs[kwarg]}")
            setattr(self, kwarg, kwargs[kwarg])

        self.validate_mode()

    # @property
    # def n(self):
    #     """
    #     returns radial degree `n` of pulsation mode
    #     :return: int
    #     """
    #     return self._n
    #
    # @n.setter
    # def n(self, radial_degree):
    #     """
    #     setter for radial degree of pulsation mode
    #     :param radial_degree: int
    #     :return:
    #     """
    #     try:
    #         self._n = np.int(radial_degree)
    #     except ValueError:
    #         raise ValueError('Value for radial degree `n`={0} in pulsation mode class instance {1} is not valid.'
    #                          .format(radial_degree, PulsationMode.__name__))

    def transform_input(self, **kwargs):
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


    @property
    def mode_axis_phi(self):
        """
        Returns longitude angle of pulsation mode axis at t_0.

        :return: (npumpy.)float; in radians
        """
        return self._mode_axis_phi

    @mode_axis_phi.setter
    def mode_axis_phi(self, mode_axis_phi):
        """
        Setter for longitude of pulsation mode axis. 
        If unit is not supplied, degrees are assumed.

        :param mode_axis_phi: Union[(numpy.)float, (numpy.)int, astropy.units.quantity.Quantity]
        :return:
        """
        if isinstance(mode_axis_phi, units.Quantity):
            self._mode_axis_phi = np.float64(mode_axis_phi.to(units.ARC_UNIT))
        elif isinstance(mode_axis_phi, (int, np.int, float, np.float)):
            self._mode_axis_phi = np.float64((mode_axis_phi * units.deg).to(units.ARC_UNIT))
        else:
            raise TypeError('Input of variable `mode_axis_phi` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if not 0 <= self._mode_axis_phi <= c.FULL_ARC:
            raise ValueError(f'Value of `mode_axis_phi`: {self._mode_axis_phi} is outside bounds (0, 2pi).')
