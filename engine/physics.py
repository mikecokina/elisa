import logging
from engine import utils

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : [%(levelname)s] : %(name)s : %(message)s')


class Physics:
    KWARGS = []
    OPTIONAL_KWARGS = ['reflection_effect', 'reflection_effect_iterations']
    ALL_KWARGS = KWARGS + OPTIONAL_KWARGS

    def __init__(self, **kwargs):
        utils.invalid_kwarg_checker(kwargs, Physics.ALL_KWARGS, Physics)

        # get logger
        self._logger = logging.getLogger(Physics.__name__)
        self._logger.info("Initialising object {}".format(Physics.__name__))

        self._logger.debug("Setting property components "
                           "of class instance {}".format(Physics.__name__))

        self._reflection_effect = None
        self._reflection_effect_iterations = None

        # check for missing kwargs
        utils.check_missing_kwargs(Physics.KWARGS, kwargs, instance_of=Physics)

        # we already ensured that all kwargs are valid and all mandatory kwargs are present so lets set class attributes
        for kwarg in kwargs:
            self._logger.debug("Setting property {} "
                               "of class instance {} to {}".format(kwarg, Physics.__name__, kwargs[kwarg]))
            setattr(self, kwarg, kwargs[kwarg])

    @property
    def reflection_effect(self):
        """
        returns value of switch that enables calculation of reflection effect

        :return: boolean
        """
        return self._reflection_effect

    @reflection_effect.setter
    def reflection_effect(self, value):
        """
        setter for reflection effect switch, use bool as its value

        :param value: boolean - if True reflection effect is implemented
        :return:
        """
        if isinstance(value, bool):
            self._reflection_effect = value
        elif isinstance(value, None):
            return
        else:
            raise TypeError('Variable `reflection_effect` has to be boolean type.')

    @property
    def reflection_effect_iterations(self):
        """
        returns number of times the reflection effect will be iteratively calculated on both components, the higher
        number, the higher precision but at expense of higher computational cost.
        :return:
        """
        return self._reflection_effect_iterations

    @reflection_effect_iterations.setter
    def reflection_effect_iterations(self, value):
        """
        setter for number of reflection effect iterations

        :param value: integer
        :return:
        """
        if self._reflection_effect is None:
            return
        elif not self._reflection_effect:
            self._logger.warning('Setting of the number of reflection effect iterations will have no effect on '
                                 'simulation because reflection effect is switched off. If you want to turn reflection '
                                 'effect on, please set the switch `reflection_effect` = True.')
        if not isinstance(value, int):
            raise TypeError('Non-integer value encountered in reflection_effect_iteration setter.')
        if value < 0:
            raise ValueError('Negative value encountered in reflection_effect_iteration setter.')
        self._reflection_effect_iterations = value
