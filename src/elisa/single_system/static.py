from elisa import const as c, units as U
from astropy import units as u


def angular_velocity(rotation_period):
    """
    rotational angular velocity of the star
    :return:
    """
    return c.FULL_ARC / (rotation_period * U.PERIOD_UNIT).to(u.s).value
