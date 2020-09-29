import os
import sys
import numpy as np
import pandas as pd

from . import utils as outils
from . plot import Plot
from . passband import PassbandContainer, init_bolometric_passband
from .. binary_system.system import BinarySystem
from .. binary_system.curves.community import RadialVelocitySystem
from .. single_system.system import SingleSystem
from .. import settings
from .. utils import is_empty
from .. logger import getLogger
from .. import (
    units as u,
    umpy as up
)

logger = getLogger('observer.observer')


class Observables(object):
    def __init__(self, observer):
        self.observer = observer

    def lc(self, from_phase=None, to_phase=None, phase_step=None, phases=None, normalize=False):
        return self.observer.lc(from_phase, to_phase, phase_step, phases, normalize)

    def rv(self, from_phase=None, to_phase=None, phase_step=None, phases=None, normalize=False, method=None):
        return self.observer.rv(from_phase, to_phase, phase_step, phases, normalize, method)


class Observer(object):
    def __init__(self, passband=None, system=None):
        """
        Initializer for observer class.

        :param passband: string; for valid filter name see settings.py file
        :param system: system instance (BinarySystem or SingleSystem)
        """
        if passband is None:
            passband = list()
        logger.info("initialising Observer instance")
        # specifying what kind of system is observed
        self._system = system
        self._system_cls = type(self._system)

        self.left_bandwidth = sys.float_info.max
        self.right_bandwidth = 0.0
        self.passband = dict()
        self.init_passband(passband)

        self._observables = ['lc', 'rv']
        self.phases = None
        self.times = None
        self.fluxes = None
        self.fluxes_unit = None
        self.radial_velocities = dict()
        self.rv_unit = None

        self.plot = Plot(self)
        self.observe = Observables(self)

    @property
    def system_cls(self):
        return self._system_cls

    @system_cls.setter
    def system_cls(self, value):
        self._system_cls = value

    def init_passband(self, passband):
        """
        Passband initializing method for Observer instance.
        During initialialization `self.passband` Dict is fill in way::

            {`passband`: PassbandContainer()}

        and global left and right passband bandwidth is set.
        If there is several passband defined on different intervals, e.g. ([350, 650], [450, 750]) then global limits
        are total boarder values, in example case as [350, 750].

        :param passband: Union[str; Iterable[str]]
        """
        passband = [passband] if isinstance(passband, str) else passband
        for band in passband:
            if band in ['bolometric']:
                psbnd, right_bandwidth, left_bandwidth = init_bolometric_passband()
            else:
                df = self.get_passband_df(band)
                left_bandwidth = df[settings.PASSBAND_DATAFRAME_WAVE].min()
                right_bandwidth = df[settings.PASSBAND_DATAFRAME_WAVE].max()
                psbnd = PassbandContainer(table=df, passband=band)

            self.setup_bandwidth(left_bandwidth=left_bandwidth, right_bandwidth=right_bandwidth)
            self.passband[band] = psbnd

    def setup_bandwidth(self, left_bandwidth, right_bandwidth):
        """
        Find whether supplied left and right bandwidth are in currently set boundaries and nothing has to be done
        or any is out of current bound and related has to be changed to higher (`right_bandwidth`)
        or lower (`left_bandwidth`).


        :param left_bandwidth: float;
        :param right_bandwidth: float;
        """
        if left_bandwidth < self.left_bandwidth:
            self.left_bandwidth = left_bandwidth
        if right_bandwidth > self.right_bandwidth:
            self.right_bandwidth = right_bandwidth

    @staticmethod
    def get_passband_df(passband):
        """
        Read content o passband table (csv file) based on passband name.

        :param passband: str;
        :return: pandas.DataFrame;
        """
        if passband not in settings.PASSBANDS:
            raise ValueError('Invalid or unsupported passband function')
        file_path = os.path.join(settings.PASSBAND_TABLES, str(passband) + '.csv')
        df = pd.read_csv(file_path)
        df[settings.PASSBAND_DATAFRAME_WAVE] = df[settings.PASSBAND_DATAFRAME_WAVE] * 10.0
        return df

    def lc(self, from_phase=None, to_phase=None, phase_step=None, phases=None, normalize=False):
        """
        Method for simulated observation. Based on input parmeters and supplied Observer system on initialization
        will compute lightcurve.

        :param normalize: bool;
        :param from_phase: float;
        :param to_phase: float;
        :param phase_step: float;
        :param phases: Iterable float;
        :return: Dict;
        """
        if phases is None and (from_phase is None or to_phase is None or phase_step is None):
            raise ValueError("Missing arguments. Specify phases.")

        if is_empty(phases):
            phases = up.arange(start=from_phase, stop=to_phase, step=phase_step)

        phases = np.array(phases) - self._system.phase_shift

        # reduce phases to only unique ones from interval (0, 1) in general case without pulsations
        base_phases, base_phases_to_origin = self.phase_interval_reduce(phases)

        logger.info(f"observation is running")
        # calculates lines of sight for corresponding phases
        position_method = self._system.get_positions_method()

        lc_kwargs = dict(
            passband=self.passband,
            left_bandwidth=self.left_bandwidth,
            right_bandwidth=self.right_bandwidth,
            atlas=settings.ATM_ATLAS,
            phases=base_phases,
            position_method=position_method
        )

        curves = self._system.compute_lightcurve(**lc_kwargs)

        # remap unique phases back to original phase interval
        for items in curves:
            curves[items] = np.array(curves[items])[base_phases_to_origin]

            # adding additional light
            correction = np.mean(curves[items]) * self._system.additional_light / (1.0 - self._system.additional_light)
            curves[items] += correction

        self.phases = phases + self._system.phase_shift
        if normalize:
            self.fluxes, _ = outils.normalize_light_curve(y_data=curves, kind='maximum', top_fraction_to_average=0.0)
            self.fluxes_unit = u.dimensionless_unscaled
        else:
            self.fluxes = curves
            self.fluxes_unit = u.W / u.m**2
        logger.info("observation finished")
        return self.phases, self.fluxes

    def rv(self, from_phase=None, to_phase=None, phase_step=None, phases=None, normalize=False, method=None):
        """
        Method for simulation of observation radial velocity curves.

        :param normalize: bool;
        :param from_phase: float;
        :param to_phase: float;
        :param phase_step: float;
        :param phases: Iterable float;
        :param method: str; method for calculation of radial velocities, `point_mass` or `radiometric`
        :return: Tuple[numpy.array, numpy.array, numpy.array]; phases, primary rv, secondary rv
        """
        method = settings.RV_METHOD if method is None else method

        if phases is None and (from_phase is None or to_phase is None or phase_step is None):
            raise ValueError("Missing arguments. Specify phases.")

        if is_empty(phases):
            phases = up.arange(start=from_phase, stop=to_phase, step=phase_step)
        phases = np.array(phases) - self._system.phase_shift

        # reduce phases to only unique ones from interval (0, 1) in general case without pulsations
        base_phases, base_phases_to_origin = self.phase_interval_reduce(phases)

        self.radial_velocities = self._system.compute_rv(
            **dict(
                phases=base_phases,
                position_method=self._system.get_positions_method(),
                method=method
            )
        )

        # remap unique phases back to original phase interval
        for items in self.radial_velocities:
            self.radial_velocities[items] = np.array(self.radial_velocities[items])[base_phases_to_origin]

        self.phases = phases + self._system.phase_shift
        self.rv_unit = u.m / u.s
        if normalize:
            self.rv_unit = u.dimensionless_unscaled
            _max = np.max([np.max(item) for item in self.radial_velocities.values()])
            self.radial_velocities = {key: value/_max for key, value in self.radial_velocities.items()}

        return self.phases, self.radial_velocities

    def phase_interval_reduce(self, phases):
        """
        Function reduces original phase interval to base interval (0, 1) in case of LC without pulsations.

        :param phases: ndarray; phases to reduce
        :return: Tuple; (base_phase: ndarray, reverse_indices: ndarray)

        ::

            base_phases:  ndarray of unique phases between (0, 1)
            reverse_indices: ndarray mask applicable to `base_phases` which will reconstruct original `phases`
        """
        if self._system_cls == BinarySystem or str(self._system_cls) == str(BinarySystem):
            # function shouldn't search for base phases if system has pulsations or is asynchronous with spots
            has_pulsation_test = self._system.primary.has_pulsations() | self._system.secondary.has_pulsations()

            test1 = (self._system.primary.synchronicity != 1.0) & self._system.primary.has_spots()
            test2 = (self._system.secondary.synchronicity != 1.0) & self._system.secondary.has_spots()
            asynchronous_spotty_test = test1 | test2

            if has_pulsation_test | asynchronous_spotty_test:
                return phases, up.arange(phases.shape[0])
            else:
                base_interval = np.round(phases % 1, 9)
                return np.unique(base_interval, return_inverse=True)

        elif self._system_cls == SingleSystem or str(self._system_cls) == str(SingleSystem):
            has_pulsation_test = self._system.star.has_pulsations()
            has_spot_test = self._system.star.has_spots()

            # the most complex case, has to be solved for each phase
            if has_pulsation_test:
                return phases, up.arange(phases.shape[0])
            # in case of just spots on surface, unique (0.1) phases are only needed
            elif has_spot_test and not has_pulsation_test:
                base_interval = np.round(phases % 1, 9)
                return np.unique(base_interval, return_inverse=True)
            # in case of clear surface wo pulsations and spots, only single observation is needed
            else:
                return np.zeros(1), np.zeros(phases.shape[0], dtype=int)

        elif self._system_cls == RadialVelocitySystem or str(self._system_cls) == str(RadialVelocitySystem):
            return phases, up.arange(phases.shape[0], dtype=np.int)

        else:
            raise NotImplemented("not implemented")
