import numpy as np
import logging
from elisa.engine import utils
from astropy import units as u
from elisa.engine import units as e_units
from elisa.engine import const as c


class PulsationMode(object):
    """
    pulsation mode data container
    """
    MANDATORY_KWARGS = ["l", "m", "amplitude", "frequency"]

    OPTIONAL_KWARGS = ["start_phase", 'mode_axis_theta', 'mode_axis_phi']
    ALL_KWARGS = MANDATORY_KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs=kwargs, kwarglist=PulsationMode.ALL_KWARGS, instance=self.__class__.__name__)

        # get logger
        self._logger = logging.getLogger(PulsationMode.__name__)
        self._logger.info(f"initialising object {self.__class__.__name__}")
        self._logger.debug(f"setting property components of class instance {self.__class__.__name__}")

        # self._n = np.nan
        self._l = np.nan
        self._m = np.nan
        self._amplitude = np.nan
        self._frequency = np.nan
        self._start_phase = 0
        self._mode_axis_theta = 0
        self._mode_axis_phi = 0

        self._logger = logging.getLogger(PulsationMode.__name__)

        utils.check_missing_kwargs(PulsationMode.MANDATORY_KWARGS, kwargs, instance_of=PulsationMode)

        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        for kwarg in kwargs:
            self._logger.debug(f"setting property {kwarg} "
                               f"of class instance {self.__class__.__name__} to {kwargs[kwarg]}")
            setattr(self, kwarg, kwargs[kwarg])

        # checking validity of parameters
        if abs(self.m) > self.l:
            raise ValueError(f'Absolute value of degree of mode m: {self.l} cannot '
                             f'be higher than non-radial order of pulsations l: {self.m}.')

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
        R eturns number of surface nodal planes.

        :return: int
        """
        return self._l

    @l.setter
    def l(self, surface_nodal_planes):
        """
        Setter for number of surface nodal planes.

        :param surface_nodal_planes: int
        :return:
        """
        try:
            self._l = np.int(surface_nodal_planes)
        except ValueError:
            raise ValueError(f'Value for number of surface nodal planes is `l`={surface_nodal_planes} '
                             f'in pulsation mode class instance {self.__class__.__name__} is not valid.')

    @property
    def m(self):
        """
        Returns number of azimutal surface nodal planes for given pulsation mode.

        :return: int
        """
        return self._m

    @m.setter
    def m(self, azimutal_nodal_planes: int):
        """
        Setter for number of azimutal nodal planes.

        :param azimutal_nodal_planes: int
        :return:
        """
        try:
            self._m = np.int(azimutal_nodal_planes)
        except ValueError:
            raise ValueError(f'Value for number of azimutal nodal planes is `m`={azimutal_nodal_planes} '
                             f'in pulsation mode class instance {self.__class__.__name__} is not valid.')

    @property
    def amplitude(self):
        """
        Returns amplitude of pulsation mode in kelvins.

        :return: float
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        """
        Setter for temperature amplitude of pulsation mode.
        
        :param amplitude: float or astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(amplitude, u.quantity.Quantity):
            self._amplitude = np.float64(amplitude.to(e_units.TEMPERATURE_UNIT))
        elif isinstance(amplitude, (int, np.int, float, np.float)):
            self._amplitude = np.float64(amplitude)
        else:
            raise TypeError('Value of `amplitude` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if self._amplitude < 0:
            raise ValueError('Temperature amplitude of mode has to be non-negative number.')

    @property
    def frequency(self):
        """
        Returns frequency of pulsation mode in default frequency unit.
        
        :return: float
        """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        """
        Frequency setter.
        If unit in astropy format is not given, default frequency unit is assumed.

        :param frequency: float or astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(frequency, u.quantity.Quantity):
            self._frequency = np.float64(frequency.to(e_units.FREQUENCY_UNIT))
        elif isinstance(frequency, (int, np.int, float, np.float)):
            self._frequency = np.float64(frequency)
        else:
            raise TypeError('Value of `frequency` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')

    @property
    def start_phase(self):
        """
        Phase shift of the pulsation mode. 
        It is basically constant that will be added to time dependent part of the equation.

        :return: float
        """
        return self._start_phase

    @start_phase.setter
    def start_phase(self, phase):
        """
        Setter for phase shift of the given pulsation mode.

        :param phase: float
        :return:
        """
        try:
            self._start_phase = np.float(phase) if phase is not None else 0.0
        except TypeError:
            raise TypeError(f'Invalid data type {type(phase)} for `start_phase` parameter for '
                            f'{self.__class__.__name__} pulsation mode instance.')

    @property
    def mode_axis_theta(self):
        """
        Returns polar latitude angle of pulsation mode axis.

        :return: (numpy.)float; in radians
        """
        return self._mode_axis_theta

    @mode_axis_theta.setter
    def mode_axis_theta(self, mode_axis_theta):
        """
        Setter for latitude of pulsation mode axis. 
        If unit is not supplied, degrees are assumed.

        :param mode_axis_theta: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(mode_axis_theta, u.quantity.Quantity):
            self._mode_axis_theta = np.float64(mode_axis_theta.to(e_units.ARC_UNIT))
        elif isinstance(mode_axis_theta, (int, np.int, float, np.float)):
            self._mode_axis_theta = np.float64((mode_axis_theta*u.deg).to(e_units.ARC_UNIT))
        else:
            raise TypeError('Input of variable `mode_axis_theta` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if not 0 <= self._mode_axis_theta < c.PI:
            raise ValueError(f'Value of `mode_axis_theta`: {self._mode_axis_theta} is outside bounds (0, pi).')

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

        :param mode_axis_phi: (numpy.)int, (numpy.)float, astropy.unit.quantity.Quantity
        :return:
        """
        if isinstance(mode_axis_phi, u.quantity.Quantity):
            self._mode_axis_phi = np.float64(mode_axis_phi.to(e_units.ARC_UNIT))
        elif isinstance(mode_axis_phi, (int, np.int, float, np.float)):
            self._mode_axis_phi = np.float64((mode_axis_phi * u.deg).to(e_units.ARC_UNIT))
        else:
            raise TypeError('Input of variable `mode_axis_phi` is not (numpy.)int or (numpy.)float '
                            'nor astropy.unit.quantity.Quantity instance.')
        if not 0 <= self._mode_axis_phi <= c.FULL_ARC:
            raise ValueError(f'Value of `mode_axis_phi`: {self._mode_axis_phi} is outside bounds (0, 2pi).')
