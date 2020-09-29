import json
import logging
import os
import os.path as op
import tempfile
import unittest
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from elisa import const, units as u, settings
from elisa import umpy as up
from elisa.base.container import StarContainer
from elisa.base.star import Star
from elisa.binary_system.container import OrbitalPositionContainer
from elisa.single_system.container import SystemContainer
from elisa.binary_system.system import BinarySystem
from elisa.single_system.system import SingleSystem
from elisa.const import Position, SinglePosition
from elisa.binary_system.orbit import orbit
from elisa.utils import is_empty
from collections.abc import Iterable

ax3 = Axes3D


def plot_points(points_1, points_2, label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    var = up.concatenate([points_1, points_2]) if not is_empty(points_2) else points_1

    xx = np.array(list(zip(*var))[0])
    yy = np.array(list(zip(*var))[1])
    zz = np.array(list(zip(*var))[2])

    scat = ax.scatter(xx, yy, zz)
    scat.set_label(label)
    ax.legend()

    max_range = np.array([xx.max() - xx.min(), yy.max() - yy.min(), zz.max() - zz.min()]).max() / 2.0

    mid_x = (xx.max() + xx.min()) * 0.5
    mid_y = (yy.max() + yy.min()) * 0.5
    mid_z = (zz.max() + zz.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


def plot_faces(points, faces, label):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.set_label(label)
    ax.legend()

    clr = 'b'
    pts = points
    fcs = faces

    plot = ax.plot_trisurf(
        pts[:, 0], pts[:, 1],
        pts[:, 2], triangles=fcs,
        antialiased=True, shade=False, color=clr)
    plot.set_edgecolor('black')

    plt.show()


def polar_gravity_acceleration(bs, component=None, components_distance=None):
    for _componet in component:
        components_instance = getattr(bs, _componet)

        mass_ratio = bs.mass_ratio if _componet == "primary" else 1.0 / bs.mass_ratio

        polar_radius = components_instance.polar_radius
        x_com = (mass_ratio * components_distance) / (1.0 + mass_ratio)
        semi_major_axis = bs.semi_major_axis

        primary_mass, secondary_mass = bs.primary.mass, bs.secondary.mass
        if _componet == "secondary":
            primary_mass, secondary_mass = secondary_mass, primary_mass

        r_vector = np.array([0.0, 0.0, polar_radius * semi_major_axis])
        centrifugal_distance = np.array([x_com * semi_major_axis, 0.0, 0.0])
        actual_distance = np.array([components_distance * semi_major_axis, 0., 0.])
        h_vector = r_vector - actual_distance
        angular_velocity = orbit.angular_velocity(bs.period, bs.eccentricity, components_distance)

        block_a = - ((const.G * primary_mass) / np.linalg.norm(r_vector) ** 3) * r_vector
        block_b = - ((const.G * secondary_mass) / np.linalg.norm(h_vector) ** 3) * h_vector
        block_c = - (angular_velocity ** 2) * centrifugal_distance

        g = block_a + block_b + block_c

        # magnitude of polar gravity acceleration in physical CGS units
        return np.linalg.norm(g) * 1e2


def prepare_binary_system(params, spots_primary=None, spots_secondary=None):
    primary = Star(mass=params["primary_mass"], surface_potential=params["primary_surface_potential"],
                   synchronicity=params["primary_synchronicity"],
                   t_eff=params["primary_t_eff"], gravity_darkening=params["primary_gravity_darkening"],
                   albedo=params['primary_albedo'],
                   metallicity=0.0, spots=spots_primary)

    secondary = Star(mass=params["secondary_mass"], surface_potential=params["secondary_surface_potential"],
                     synchronicity=params["secondary_synchronicity"],
                     t_eff=params["secondary_t_eff"], gravity_darkening=params["secondary_gravity_darkening"],
                     albedo=params['secondary_albedo'],
                     metallicity=0.0, spots=spots_secondary)

    return BinarySystem(primary=primary,
                        secondary=secondary,
                        argument_of_periastron=params["argument_of_periastron"],
                        gamma=params["gamma"],
                        period=params["period"],
                        eccentricity=params["eccentricity"],
                        inclination=params["inclination"],
                        primary_minimum_time=params["primary_minimum_time"],
                        phase_shift=params["phase_shift"])


def prepare_star(star_params):
    return Star(**star_params)


def prepare_orbital_position_container(system):
    orbital_position_container = OrbitalPositionContainer(
        primary=StarContainer.from_properties_container(system.primary.to_properties_container()),
        secondary=StarContainer.from_properties_container(system.secondary.to_properties_container()),
        position=Position(*(0, 1.0, 0.0, 0.0, 0.0)),
        **system.properties_serializer()
    )
    return orbital_position_container


def prepare_single_system(params, spots=None, pulsations=None):
    star = Star(mass=params['mass'], t_eff=params['t_eff'],
                gravity_darkening=params['gravity_darkening'],
                polar_log_g=params['polar_log_g'],
                metallicity=0.0,
                spots=spots, pulsations=pulsations)

    return SingleSystem(star=star,
                        gamma=params["gamma"],
                        inclination=params["inclination"],
                        rotation_period=params['rotation_period'])


def prepare_single_system_container(system):
    system_container = SystemContainer(
        star=StarContainer.from_properties_container(system.star.to_properties_container()),
        position=SinglePosition(*(0, 0.0, 0.0)),
        **system.properties_serializer()
    )
    return system_container


def normalize_lc_for_unittests(flux_arr):
    return np.array(flux_arr) / max(flux_arr)


def normalize_lv_for_unittests(primary, secondary):
    _max = np.max(np.abs([primary, secondary]))
    primary /= _max
    secondary /= _max
    return primary, secondary


def dump_lc_observation_to_json(observation, filename):
    observation = list(observation)
    observation[0] = observation[0].tolist()
    observation[1] = {filter: item.tolist() for filter, item in observation[1].items()}
    with open(filename, 'w') as outfile:
        json.dump(observation, outfile, indent=4)


def dump_rv_observation_to_json(observation, filename):
    dct = {'phases': list(observation[0]), 'primary': list(observation[1]['primary']),
           'secondary': list(observation[1]['secondary'])}
    with open(filename, 'w') as outfile:
        json.dump(dct, outfile, indent=4)


def load_light_curve(filename):
    path = op.join(op.dirname(op.abspath(__file__)), "data", "light_curves", "curves", filename)
    with open(path, "r") as f:
        content = f.read()
        return json.loads(content)


def load_radial_curve(filename):
    path = op.join(op.dirname(op.abspath(__file__)), "data", "radial_curves", "curves", filename)
    with open(path, "r") as f:
        content = f.read()
        return json.loads(content)


def find_indices_of_duplicates(records_array):
    """
    returns duplicate values and indices of duplicates
    :param records_array: np.array;
    :return: tuple; duplicit values, corresponding indices (iterator)
    """
    idx_sort = np.argsort(records_array)
    sorted_records_array = records_array[idx_sort]
    vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                       return_index=True)

    # sets of indices
    res = np.split(idx_sort, idx_start[1:])
    # filter them with respect to their size, keeping only items occurring more than once

    vals = vals[count > 1]
    res = filter(lambda x: x.size > 1, res)
    return vals, res


def surface_closed(faces, points):
    """
    tests if surface given by `points` and `faces` contains all points, is closed, and without overlaps

    :param faces: np.array;
    :param points: np.array
    :return: bool;
    """
    # removing duplicite points on borders between components and/or spots
    unique_face_vertices, inverse_face_indices = np.unique(faces, return_inverse=True)
    # if this will not pass, there are points not included in surface
    if (unique_face_vertices != np.arange(unique_face_vertices.shape[0])).all():
        return False
    points_from_uniq_vertices = points[unique_face_vertices]
    # renormalizing unique surface points so we can round them to specific precision
    points_from_uniq_vertices = np.round(points_from_uniq_vertices / np.max(np.abs(points_from_uniq_vertices)), 6)
    # filtering out duplicit points on borders
    _, inverse_point_indices = np.unique(points_from_uniq_vertices, return_inverse=True, axis=0)

    # re-routing indices of duplicit vertices to index of unique point so the code below will work properly
    vals, duplicit_idx_iterator = find_indices_of_duplicates(inverse_point_indices)
    for duplicit_idx in duplicit_idx_iterator:
        unique_face_vertices[duplicit_idx] = unique_face_vertices[duplicit_idx[0]]

    faces = unique_face_vertices[inverse_face_indices].reshape(faces.shape)

    edges = np.row_stack((np.column_stack((faces[:, 0], faces[:, 1])),
                          np.column_stack((faces[:, 1], faces[:, 2])),
                          np.column_stack((faces[:, 2], faces[:, 0]))))
    for edge in edges:
        in_array = np.isin(element=faces, test_elements=edge)
        occurences = np.sum(in_array, axis=1)  # searching for particular edge
        edge_is_in_count = np.sum(occurences == 2)
        # every edge should belong to exactly two faces
        if edge_is_in_count != 2:
            return False

    return True


class ElisaTestCase(unittest.TestCase):
    CONFIG_FILE = op.join(tempfile.gettempdir(), "elisa.ini")

    def touch_default_config(self):
        with open(self.CONFIG_FILE, "w") as f:
            f.write("")

    def write_default_support(self, ld_tables, atm_tables):
        # because of stupid windows MP implementation
        content = f'[support]\n' \
                  f'ld_tables={ld_tables}\n' \
                  f'castelli_kurucz_04_atm_tables={atm_tables}\n\n'
        with open(self.CONFIG_FILE, "w") as f:
            f.write(content)

    def setUpClass(*args, **kwargs):
        logging.disable(logging.CRITICAL)
        # logging.disable(logging.NOTSET)
        pass

    def setUp(self):
        os.environ["ELISA_CONFIG"] = self.CONFIG_FILE
        settings.configure(**settings.DEFAULT_SETTINGS)
        self.touch_default_config()
        settings.configure(**{"CONFIG_FILE": ElisaTestCase.CONFIG_FILE})

    def tearDown(self):
        settings.configure(**settings.DEFAULT_SETTINGS)
        if op.isfile(self.CONFIG_FILE):
            os.remove(self.CONFIG_FILE)


BINARY_SYSTEM_PARAMS = {
    "detached": {
        "primary_mass": 2.0, "secondary_mass": 1.0,
        "primary_surface_potential": 100.0, "secondary_surface_potential": 100.0,
        "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
        "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 8,
        "eccentricity": 0.0, "inclination": const.HALF_PI * u.deg, "primary_minimum_time": 0.0,
        "phase_shift": 0.0,
        "primary_t_eff": 5000, "secondary_t_eff": 5000,
        "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        "primary_albedo": 0.6, "secondary_albedo": 0.6,
    },  # compact spherical components on circular orbit

    "detached-physical": {
        "primary_mass": 2.0, "secondary_mass": 1.0,
        "primary_surface_potential": 15.0, "secondary_surface_potential": 15.0,
        "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
        "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 5.0,
        "eccentricity": 0.0, "inclination": const.HALF_PI * u.deg, "primary_minimum_time": 0.0,
        "phase_shift": 0.0,
        "primary_t_eff": 5000, "secondary_t_eff": 5000,
        "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        "primary_albedo": 0.6, "secondary_albedo": 0.6,
    },  # compact spherical components on circular orbit

    "detached.ecc": {
        "primary_mass": 2.0, "secondary_mass": 1.0,
        "primary_surface_potential": 4.8, "secondary_surface_potential": 4.0,
        "primary_synchronicity": 1.5, "secondary_synchronicity": 1.2,
        "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
        "eccentricity": 0.3, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
        "phase_shift": 0.0,
        "primary_t_eff": 5000, "secondary_t_eff": 5000,
        "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        "primary_albedo": 0.6, "secondary_albedo": 0.6,
    },  # close tidally deformed components with asynchronous rotation on eccentric orbit

    "over-contact": {
        "primary_mass": 2.0, "secondary_mass": 1.0,
        "primary_surface_potential": 2.7,
        "secondary_surface_potential": 2.7,
        "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
        "argument_of_periastron": 90 * u.deg, "gamma": 0.0, "period": 1.0,
        "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
        "phase_shift": 0.0,
        "primary_t_eff": 5000, "secondary_t_eff": 5000,
        "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        "primary_albedo": 0.6, "secondary_albedo": 0.6,
    },  # over-contact system

    "semi-detached": {
        "primary_mass": 2.0, "secondary_mass": 1.0,
        "primary_surface_potential": 2.875844632141054,
        "secondary_surface_potential": 2.875844632141054,
        "primary_synchronicity": 1.0, "secondary_synchronicity": 1.0,
        "argument_of_periastron": const.HALF_PI * u.rad, "gamma": 0.0, "period": 1.0,
        "eccentricity": 0.0, "inclination": 90.0 * u.deg, "primary_minimum_time": 0.0,
        "phase_shift": 0.0,
        "primary_t_eff": 5000, "secondary_t_eff": 5100,
        "primary_gravity_darkening": 1.0, "secondary_gravity_darkening": 1.0,
        "primary_albedo": 0.6, "secondary_albedo": 0.6,
    }
}

SINGLE_SYSTEM_PARAMS = {
    "spherical": {
        "mass": 1.0,
        "t_eff": 5774 * u.K,
        "gravity_darkening": 0.32,
        "polar_log_g": 4.1,
        "gamma": 0.0,
        "inclination": 90.0 * u.deg,
        "rotation_period": 30 * u.d,
    },
    "squashed": {
        "mass": 1.0,
        "t_eff": 5774 * u.K,
        "gravity_darkening": 0.32,
        "polar_log_g": 4.1,
        "gamma": 0.0,
        "inclination": 90.0 * u.deg,
        "rotation_period": 0.3818 * u.d,
    },
}

SOLAR_MODEL = {
    "mass": 1.0,
    "t_eff": 5772 * u.K,
    "gravity_darkening": 0.32,
    "polar_log_g": 4.43775 * u.dex(u.cm / u.s ** 2),
    "gamma": 0.0,
    # "inclination": 82.5 * u.deg,
    "inclination": 90.0 * u.deg,
    "rotation_period": 25.38 * u.d,
}

SPOTS_META = {
    "primary":
        [
            {"longitude": 90,
             "latitude": 58,
             "angular_radius": 35,
             "temperature_factor": 0.95},
        ],

    "secondary":
        [
            {"longitude": 60,
             "latitude": 45,
             "angular_radius": 28,
             "temperature_factor": 0.9},
        ]
}

SPOTS_OVERLAPPED = [
    {"longitude": 90,
     "latitude": 60,
     "angular_radius": 15,
     "temperature_factor": 0.98},
    {"longitude": 90,
     "latitude": 58,
     "angular_radius": 25,
     "temperature_factor": 0.99},
]

SPOT_TO_RAISE = [
    {"longitude": 60,
     "latitude": 45,
     "angular_radius": 28,
     "temperature_factor": 0.1},
]

IDENTICAL_BINARY = {
  "system": {
    "inclination": 80.0,
    "period": 2.0,
    "argument_of_periastron": 170,
    "gamma": 0.0,
    "eccentricity": 0.00,
    "primary_minimum_time": 0.0,
    "phase_shift": 0.0
  },
  "primary": {
    "mass": 2.0,
    "surface_potential": 7.0,
    "synchronicity": 1.0,
    "t_eff": 7000.0,
    "gravity_darkening": 1.0,
    "albedo": 1.0,
    "metallicity": 0.0
  },
  "secondary": {
    "mass": 2.0,
    "surface_potential": 7.0,
    "synchronicity": 1.0,
    "t_eff": 7000.0,
    "gravity_darkening": 1.0,
    "albedo": 1.0,
    "metallicity": 0.0
  }
}

APPROX_SETTINGS = \
    {"no_approx":
         {"POINTS_ON_ECC_ORBIT": -1, "MAX_RELATIVE_D_R_POINT": 0.0, "MAX_SUPPLEMENTAR_D_DISTANCE": 0.0},
     "approx_one":
         {"POINTS_ON_ECC_ORBIT": 1, "MAX_RELATIVE_D_R_POINT": 0.003, "MAX_SUPPLEMENTAR_D_DISTANCE": 0.001},
     "approx_two":
         {"POINTS_ON_ECC_ORBIT": int(1e6), "MAX_RELATIVE_D_R_POINT": 0.003, "MAX_SUPPLEMENTAR_D_DISTANCE": 0.001},
     "approx_three":
         {"POINTS_ON_ECC_ORBIT": int(1e6), "MAX_RELATIVE_D_R_POINT": 0.003, "MAX_SUPPLEMENTAR_D_DISTANCE": 0.0}
     }


def cutoff_float(x, keep_n):
    if isinstance(x, Iterable):
        return [float(format(x, f".{keep_n}")) for _x in x]
    return float(format(x, f".{keep_n}"))
