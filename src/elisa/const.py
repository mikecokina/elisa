import numpy as np
from collections import namedtuple


# DO NOT CHANGE ANYTHING

FULL_ARC = np.pi * 2
PI = np.pi
HALF_PI = np.pi / 2
G = 6.67408e-11
S_BOLTZMAN = 5.670373e-8
PLANCK_CONST = 6.626070150e-34
C = 299792458
SPEED_OF_LIGHT = C
BOLTZMAN_CONST = 1.380649e-23
IDEAL_ADIABATIC_GRADIENT = 0.4

AU = 149597870700.0

# global SOLAR_RADIUS
SOLAR_RADIUS = 6.955E8  # m

# global SOLAR_MASS
SOLAR_MASS = 1.9891E30  # kg

# global TEMPERATURE_LIST_LD
TEMPERATURE_LIST_LD = [3500.0, 3750.0, 4000.0, 4250.0, 4500.0, 4750.0, 5000.0, 5250.0, 5500.0, 5750.0,
                       6000.0, 6250.0, 6500.0, 6750.0, 7000.0, 7250.0, 7500.0, 7750.0, 8000.0, 8250.0,
                       8500.0, 8750.0, 9000.0, 9250.0, 9500.0, 9750.0, 10000.0, 10250.0, 10500.0, 10750.0,
                       11000.0, 11250.0, 11500.0, 11750.0, 12000.0, 12250.0, 12500.0, 12750.0, 13000.0,
                       14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
                       23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 31000.0,
                       32000.0, 33000.0, 34000.0, 35000.0, 36000.0, 37000.0, 37500.0, 38000.0, 39000.0,
                       40000.0, 41000.0, 42000.0, 42500.0, 43000.0, 44000.0, 45000.0, 46000.0, 47000.0,
                       47500.0, 48000.0, 49000.0, 50000.0]

# global GRAVITY_LIST_LD
GRAVITY_LIST_LD = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# global METALLICITY_LIST_LD
METALLICITY_LIST_LD = [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.3, -0.2, -0.1, 0.0,
                       0.1, 0.2, 0.3, 0.5, 1.0]

# CK04 atlas TEMPERATUREs
CK_TEMPERATURE_LIST_ATM = [3500.0, 3750.0, 4000.0, 4250.0, 4500.0, 4750.0, 5000.0, 5250.0, 5500.0, 5750.0, 6000.0,
                           6250.0, 6500.0, 6750.0, 7000.0, 7250.0, 7500.0, 7750.0, 8000.0, 8250.0, 8500.0, 8750.0,
                           9000.0, 9250.0, 9500.0, 9750.0, 10000.0, 10250.0, 10500.0, 10750.0, 11000.0, 11250.0,
                           11500.0, 11750.0, 12000.0, 12250.0, 12500.0, 12750.0, 13000.0, 14000.0, 15000.0, 16000.0,
                           17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0, 23000.0, 24000.0, 25000.0, 26000.0,
                           27000.0, 28000.0, 29000.0, 30000.0, 31000.0, 32000.0, 33000.0, 34000.0, 35000.0, 36000.0,
                           37000.0, 38000.0, 39000.0, 40000.0, 41000.0, 42000.0, 43000.0, 44000.0, 45000.0, 46000.0,
                           47000.0, 48000.0, 49000.0, 50000.0]

# CK04 atlas METALLICITIEs
CK_METALLICITY_LIST_ATM = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.2, 0.5]

# CK04 atlas GRAVITIEs
CK_GRAVITY_LIST_ATM = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# K93 atlas METALLICITIEs
K_METALLICITY_LIST_ATM = [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0,
                          -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0]

# K93 atlas TEMPERATUREs
K_TEMPERATURE_LIST_ATM = [3500.0, 3750.0, 4000.0, 4250.0, 4500.0, 4750.0, 5000.0, 5250.0, 5500.0, 5750.0, 6000.0,
                          6250.0, 6500.0, 6750.0, 7000.0, 7250.0, 7500.0, 7750.0, 8000.0, 8250.0, 8500.0, 8750.0,
                          9000.0, 9250.0, 9500.0, 9750.0, 10000.0, 10500.0, 11000.0, 11500.0, 12000.0, 12500.0,
                          13000.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
                          23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 31000.0, 32000.0,
                          33000.0, 34000.0, 35000.0, 37500.0, 40000.0, 42500.0, 45000.0, 47500.0, 50000.0]

# K93 atlas GRAVITIEs
K_GRAVITY_LIST_ATM = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

LEFT_BANDWIDTH_SHIFT = 100  # nm
RIGHT_BANDWIDTH_SHIFT = 100  # nm

# normalized vector pointing to the observer
LINE_OF_SIGHT = np.array([-1.0, 0.0, 0.0])

FALSE_FACE_PLACEHOLDER = np.array([-1, -1, -1])

MAX_USABLE_FLOAT = np.finfo(float).max * np.finfo(float).eps

# distance - distance between components (if applicable)
Position = namedtuple('Position', ['idx', 'distance', 'azimuth', 'true_anomaly', 'phase'])

# constant for incrased spacing of points on the seams
SEAM_CONST = 1.08
