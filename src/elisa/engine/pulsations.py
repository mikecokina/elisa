import numpy as np
import logging
from elisa.engine import utils
from astropy import units as u
from elisa.engine import units as U
from elisa.engine import const as c


class PulsationMode(object):
    """
    pulsation mode data container
    """
    KWARGS = ["l", "m", "amplitude", "frequency"]

    OPTIONAL_KWARGS = ["start_phase", 'mode_axis_theta', 'mode_axis_phi']
    ALL_KWARGS = KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs=kwargs, kwarglist=PulsationMode.ALL_KWARGS, instance=PulsationMode)

        # get logger
        self._logger = logging.getLogger(PulsationMode.__name__)
        self._logger.info("Initialising object {}".format(PulsationMode.__name__))

        self._logger.debug("Setting property components "
                           "of class instance {}".format(PulsationMode.__name__))

        # self._n = None
        self._l = None
        self._m = None
        self._amplitude = None
        self._frequency = None
        self._start_phase = 0
        self._mode_axis_theta = 0
        self._mode_axis_phi = 0

        self._logger = logging.getLogger(PulsationMode.__name__)

        utils.check_missing_kwargs(PulsationMode.KWARGS, kwargs, instance_of=PulsationMode)

        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        for kwarg in kwargs:
            self._logger.debug("Setting property {} "
                               "of class instance {} to {}".format(kwarg, PulsationMode.__name__, kwargs[kwarg]))
            setattr(self, kwarg, kwargs[kwarg])

        # checking validity of parameters
        if abs(self.m) > self.l:
            raise ValueError('Absolute value of degree of mode m: {0} cannot be higher than non-radial order of '
                             'pulsations l: {1}. '.format(self.l, self.m))

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

    @property
    def l(self):
        """
        returns number of surface nodal planes
        :return: int
        """
        return self._l

    @l.setter
    def l(self, surface_nodal_planes):
        """
        setter for number of surface nodal planes
        :param surface_nodal_planes: int
        :return:
        """
        try:
            self._l = np.int(surface_nodal_planes)
        except ValueError:
            raise ValueError('Value for number of surface nodal planes is `l`={0} in pulsation mode class instance {1} '
                             'is not valid.'.format(surface_nodal_planes, PulsationMode.__name__))

    @property
    def m(self):
        """
        returns number of azimutal surface nodal planes for given pulsation mode
        :return:
        """
        return self._m

    @m.setter
    def m(self, azimutal_nodal_planes):
        """
        setter for number of azimutal nodal planes
        :param azimutal_nodal_planes: int
        :return:
        """
        try:
            self._m = np.int(azimutal_nodal_planes)
        except:
            raise ValueError('Value for number of azimutal nodal planes is `m`={0} in pulsation mode class instance '
                             '{1} is not valid.'.format(azimutal_nodal_planes, PulsationMode.__name__))

    @property
    def amplitude(self):
        """
        returns amplitude of pulsation mode in kelvins
        :return: float
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        """
        setter for temperature amplitude of pulsation mode
        :param amplitude: float or astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(amplitude, u.quantity.Quantity):
            self._amplitude = np.float64(amplitude.to(U.TEMPERATURE_UNIT))
        elif isinstance(amplitude, (int, np.int, float, np.float)):
            self._amplitude = np.float64(amplitude)
        else:
            raise TypeError('Value of `amplitude` is not (np.)int or (np.)float nor astropy.unit.quantity.Quantity '
                            'instance.')
        if self._amplitude < 0:
            raise ValueError('Temperature amplitude of mode has to be non-negative number.')

    @property
    def frequency(self):
        """
        returns frequency of pulsation mode in default frequency unit
        :return: float
        """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        """
        frequency setter, if unit in astropy format is not given, default frequency unit is assumed

        :param frequency: float or astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(frequency, u.quantity.Quantity):
            self._frequency = np.float64(frequency.to(U.FREQUENCY_UNIT))
        elif isinstance(frequency, (int, np.int, float, np.float)):
            self._frequency = np.float64(frequency)
        else:
            raise TypeError('Value of `frequency` is not (np.)int or (np.)float nor astropy.unit.quantity.Quantity '
                            'instance.')

    @property
    def start_phase(self):
        """
        phase shift of the pulsation mode, basically constant that will be added to time dependent part of the equation.

        :return: float
        """
        return self._start_phase

    @start_phase.setter
    def start_phase(self, phase):
        """
        setter for phase shift of the given pulsation mode,

        :param phase: float
        :return:
        """
        try:
            self._start_phase = np.float(phase) if phase is not None else 0
        except TypeError:
            raise TypeError('Invalid data type {0} for `start_phase` parameter for {1} pulsation mode '
                            'instance.'.format(type(phase), PulsationMode.__name__))

    @property
    def mode_axis_theta(self):
        """
        returns polar latitude angle of pulsation mode axis

        :return: np.float - in radians
        """
        return self._mode_axis_theta

    @mode_axis_theta.setter
    def mode_axis_theta(self, mode_axis_theta):
        """
        setter for latitude of pulsation mode axis, if unit is not supplied, degrees are assumed

        :param mode_axis_theta: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(mode_axis_theta, u.quantity.Quantity):
            self._mode_axis_theta = np.float64(mode_axis_theta.to(U.ARC_UNIT))
        elif isinstance(mode_axis_theta, (int, np.int, float, np.float)):
            self._mode_axis_theta = np.float64((mode_axis_theta*u.deg).to(U.ARC_UNIT))
        else:
            raise TypeError('Input of variable `mode_axis_theta` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if not 0 <= self._mode_axis_theta < c.PI:
            raise ValueError('Value of `mode_axis_theta`: {} is outside bounds (0, pi).'.format(self._mode_axis_theta))

    @property
    def mode_axis_phi(self):
        """
        returns longitude angle of pulsation mode axis at t_0

        :return: np.float - in radians
        """
        return self._mode_axis_phi

    @mode_axis_phi.setter
    def mode_axis_phi(self, mode_axis_phi):
        """
        setter for longitude of pulsation mode axis, if unit is not supplied, degrees are assumed

        :param mode_axis_phi: (np.)int, (np.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(mode_axis_phi, u.quantity.Quantity):
            self._mode_axis_phi = np.float64(mode_axis_phi.to(U.ARC_UNIT))
        elif isinstance(mode_axis_phi, (int, np.int, float, np.float)):
            self._mode_axis_phi = np.float64((mode_axis_phi * u.deg).to(U.ARC_UNIT))
        else:
            raise TypeError('Input of variable `mode_axis_phi` is not (np.)int or (np.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if not 0 <= self._mode_axis_phi <= c.FULL_ARC:
            raise ValueError('Value of `mode_axis_phi`: {} is outside bounds (0, 2pi).'.format(self._mode_axis_phi))
