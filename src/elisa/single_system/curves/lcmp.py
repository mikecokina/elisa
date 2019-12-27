import numpy as np

from elisa.conf import config
from elisa.single_system import utils as ssutils

LD_LAW_CFS_COLUMNS = config.LD_LAW_CFS_COLUMNS[config.LIMB_DARKENING_LAW]


def compute_non_pulsating_lightcurve(*args):
    single, initial_system, phase_batch, normal_radiance, ld_cfs, kwargs = args
    position_method = kwargs.pop("position_method")

    rotational_motion = position_method(input_argument=phase_batch, return_nparray=False, calculate_from='phase')
    azimuths = [position.azimuth for position in rotational_motion]
    band_curves = {key: np.zeros(phase_batch.shape) for key in kwargs["passband"].keys()}

    for pos_idx, position in enumerate(rotational_motion):
        on_pos = ssutils.move_sys_onpos(initial_system, position)
        star_container = on_pos.star
        # dict of components
        stars = {'star': getattr(on_pos, 'star')}

        # visible_faces = np.unique(star_container.faces[star_container.indices])
        pass