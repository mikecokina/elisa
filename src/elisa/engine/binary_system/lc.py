from elisa.conf import config
from elisa.engine import atm


def compute_circular_synchronous_lightcurve(self, **kwargs):
    _temperature = [
        5551.36,
        5552.25,
        6531.81,
        7825.66,
        4500,
        19874.85,
    ]

    _metallicity = 0.11

    _logg = [
        4.12,
        3.92,
        2.85,
        2.99,
        1.1,
        3.11,
    ]

    # compute on filtered atmospheres (doesn't meeter how will be filtered)
    primary_radiance = \
        atm.NaiveInterpolatedAtm.radiance(_temperature, _logg, self.primary.metallicity, config.ATM_ATLAS, **kwargs)


if __name__ == "__main__":
    pass

