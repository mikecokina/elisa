import numpy as np
from copy import copy

import scipy

from engine import utils, const
from engine import const as c
from astropy import units as u
from engine import units as U


def angular_velocity(rotation_period):
    """
    rotational angular velocity of the star
    :return:
    """
    return c.FULL_ARC / (rotation_period * U.PERIOD_UNIT).to(u.s).value