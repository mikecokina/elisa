import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from elisa import utils, umpy as up
from elisa import const
from queue import Queue

from unittests.utils import ElisaTestCase


class TestElisaEngineUtils(ElisaTestCase):
    def test_polar_to_cartesian(self):
        radius = np.array([0.5, 0.3, 6.1])
        phi = np.array([0.0, const.PI, const.PI / 2.0])

        xs, ys = utils.polar_to_cartesian(radius, phi)
        xs, ys = np.round(xs, 4), np.round(ys, 4)

        expected_xs, expected_ys = [0.5, -0.3, 0.0], [0.0, 0.0, 6.1]
        assert_array_equal(expected_xs, xs)
        assert_array_equal(expected_ys, ys)

        radius = 0.33
        phi = const.PI / 2.0
        x, y = utils.polar_to_cartesian(radius, phi)
        assert_array_equal([round(x, 4), round(y, 4)], [0.0, 0.33])

    def test_invalid_kwarg_checker(self):
        class MockClass(object):
            kwargs = ["a", "b", "c"]

        utils.invalid_kwarg_checker(["a", "b"], MockClass.kwargs, MockClass)
        utils.invalid_kwarg_checker(["b", "c", "a"], MockClass.kwargs, MockClass)

        with self.assertRaises(Exception) as context:
            utils.invalid_kwarg_checker(["a", "x", "b"], MockClass.kwargs, MockClass)
        self.assertTrue('Invalid keyword argument' in str(context.exception))

    def test_is_plane(self):
        given, expected = 'yx', 'xy'
        self.assertTrue(utils.is_plane(given=given, expected=expected))

        given, expected = 'xy', 'xy'
        self.assertTrue(utils.is_plane(given=given, expected=expected))

        given, expected = 'xy', 'xz'
        self.assertFalse(utils.is_plane(given=given, expected=expected))

    def test_find_nearest_dist_3d(self):
        points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 1.0], [0.0, 0.3, 0.0]])
        nearest = utils.find_nearest_dist_3d(points)
        self.assertEqual(0.3, round(nearest, 4))

    def test_cartesian_to_spherical(self):
        input1 = np.array([[1., 0., 0.], [0., 0., 1.]])
        input2 = [1, 0, 0]

        expected_output1 = np.array([[1, 0, const.PI/2], [1, 0, 0]])
        expected_output2 = np.array([1, 0, const.PI/2])
        expected_output3 = np.array([1, 0, 90])

        output1 = utils.cartesian_to_spherical(input1)
        output2 = utils.cartesian_to_spherical(input2)
        output3 = utils.cartesian_to_spherical(input2, degrees=True)

        assert_array_equal(output1, expected_output1)
        assert_array_equal(output2, expected_output2)
        assert_array_equal(output3, expected_output3)

    def test_spherical_to_cartesian(self):
        input1 = np.array([[1, 0, const.PI / 2], [1, 0, 0]])
        input2 = np.array([1, 0, const.PI / 2])

        expected_output1 = np.array([[1., 0., 0.], [0., 0., 1.]])
        expected_output2 = [1, 0, 0]

        output1 = utils.spherical_to_cartesian(input1)
        output2 = utils.spherical_to_cartesian(input2)

        assert_array_equal(np.round(output1, 4), expected_output1)
        assert_array_equal(np.round(output2, 4), expected_output2)

    def test_cylindrical_to_cartesian(self):
        cylindrical = np.array([[1.3, 1.34, 4], [2, -2.5, 3], [-2.0, -1555.5, -3.1]])
        obtained = np.round(utils.cylindrical_to_cartesian(cylindrical), 4)
        expected = [[0.2974, 1.2655, 4], [-1.6023, -1.1969, 3], [1.8329, -0.8002, -3.1]]
        assert_array_equal(obtained, expected)

    @staticmethod
    def subtest_arbitrary_rotation(angle, vectors, degree, expected):
        omega = np.array([-1.0, 1.0, 1.0])
        obtained = np.round(utils.arbitrary_rotation(angle, omega, vectors, degrees=degree, omega_normalized=False), 4)
        assert_array_equal(obtained, expected)

    def test_arbitrary_rotation(self):
        to_rotate = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, 1.3, -2.1]])
        expected = np.array([[0.3333, 0.244, -0.9107], [-0.3333, 1.488, -0.8214], [-2.0297, -1.7231, -0.1065]])
        self.subtest_arbitrary_rotation(const.PI / 2.0, to_rotate, False, expected)
        self.subtest_arbitrary_rotation(90, to_rotate, True, expected)
        self.subtest_arbitrary_rotation(-270, to_rotate, True, expected)
        self.subtest_arbitrary_rotation(-3. * const.PI / 2.0, to_rotate, False, expected)

    @staticmethod
    def subtest_around_axis_rotation(angle, vectors, axis, degree, expected):
        obtained = np.round(utils.around_axis_rotation(angle, vectors, axis, degrees=degree), 4)
        assert_array_equal(expected, obtained)

    def test_around_axis_rotation(self):
        to_rotate = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, 1.3, -2.1]])

        # z axis
        expected = [[0.0, 1.0, 0.0], [-1.0, 1.0, 1.0], [-1.3, -1.0, -2.1]]
        self.subtest_around_axis_rotation(const.PI / 2.0, to_rotate, "z", False, expected)
        self.subtest_around_axis_rotation(90, to_rotate, "z", True, expected)
        self.subtest_around_axis_rotation(-270, to_rotate, "z", True, expected)

        # x axis
        expected = [[1., 0., 0.], [1., -1., 1.], [-1., 2.1, 1.3]]
        self.subtest_around_axis_rotation(const.PI / 2.0, to_rotate, "x", False, expected)
        self.subtest_around_axis_rotation(90, to_rotate, "x", True, expected)
        self.subtest_around_axis_rotation(-270, to_rotate, "x", True, expected)

        # y axis
        expected = [[0., 0., -1.], [1.,  1., -1.], [-2.1, 1.3, 1.]]
        self.subtest_around_axis_rotation(const.PI / 2.0, to_rotate, "y", False, expected)
        self.subtest_around_axis_rotation(90, to_rotate, "y", True, expected)
        self.subtest_around_axis_rotation(-270, to_rotate, "y", True, expected)

    def test_remap(self):
        x = [[3, 4, 5], [6, 7, 8], [0, 2, 1]]
        mapper = [7, 8, 9, 1, 2, 3, 4, 5, 6]
        obtained = utils.remap(x, mapper)
        expected = [[1, 2, 3], [4, 5, 6], [7, 9, 8]]
        assert_array_equal(obtained, expected)

    def test_poly_areas(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5]

        ], dtype=float)

        triangles = np.array([[0, 1, 2], [2, 3, 0], [0, 4, 5]], dtype=int)
        obtained = np.round(utils.poly_areas(points[triangles]), 4)
        expected = [0.5, 0.25, 0.1768]
        assert_array_equal(obtained, expected)

    def test_triangle_areas(self):
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5]

        ], dtype=float)

        triangles = np.array([[0, 1, 2], [2, 3, 0], [0, 4, 5]], dtype=int)
        obtained = np.round(utils.triangle_areas(triangles, points), 4)
        expected = [0.5, 0.25, 0.1768]
        assert_array_equal(obtained, expected)

    def test_calculate_distance_matrix(self):
        points1 = np.array([[0.0, 0.0], [0.0, 1.5], [1.3, -1.2]])
        points2 = np.array([[1.5, 0.0], [0.0, 0.3]])

        v, d = utils.calculate_distance_matrix(points1, points2, False)
        expected = [[1.5, 0.3],
                    [2.1213, 1.2],
                    [1.2166, 1.9849]]
        obtained = np.round(v, 4)
        assert_array_equal(expected, obtained)
        self.assertIsNone(d)

        _, d = utils.calculate_distance_matrix(points1, points2, True)
        d = np.round(d, 4)
        expected = [
            [[1., 0.], [0., 1.]],
            [[0.7071, -0.7071], [0., -1.]],
            [[0.1644, 0.9864], [-0.6549, 0.7557]]
        ]
        assert_array_equal(d, expected)

    def test_find_face_centres(self):
        faces = np.array([
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0]
            ],
            [
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0]
            ]
        ])

        obtained = np.round(utils.find_face_centres(faces), 4)
        expected = [[0.6667, 0.3333], [0.3333, 0.6667]]
        assert_array_equal(obtained, expected)

    def test_check_missing_kwargs(self):
        class MockClass(object):
            pass

        required = ["a", "b", "c"]
        supplied = ["c", "a"]

        with self.assertRaises(Exception) as context:
            utils.check_missing_kwargs(required, supplied, MockClass)
        self.assertTrue('Missing argument' in str(context.exception))

        supplied = ["c", "a", "b"]
        utils.check_missing_kwargs(required, supplied, MockClass)

    def test_numeric_logg_to_string(self):
        logg = [0.5, 0.0, 1.0, 10.0]
        obtained = [utils.numeric_logg_to_string(m) for m in logg]
        expected = ['g05', 'g00', 'g10', 'g100']
        assert_array_equal(obtained, expected)

    def test_numeric_metallicity_to_string(self):
        metalicity = [0.5, 0.0, 1.0, -1.1, 10.0]
        obtained = [utils.numeric_metallicity_to_string(m) for m in metalicity]
        expected = ['p05', 'p00', 'p10', 'm11', 'p100']
        assert_array_equal(obtained, expected)

    def test_find_nearest_value_as_matrix(self):
        look_in = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        look_for = np.array([-0.5, 2.1, -20, 5.1, 5.0])
        arr, indices = utils.find_nearest_value_as_matrix(look_in, look_for)
        exp_arr = [-1., 2., -1., 5., 5.]
        exp_indices = [0, 3, 0, 6, 6]
        assert_array_equal(arr, exp_arr)
        assert_array_equal(indices, exp_indices)

    def test_find_surrounded_as_matrix(self):
        look_in = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        look_for = np.array([-0.5, 2.1])
        obtained = utils.find_surrounded_as_matrix(look_in, look_for)
        expected = [[-1., 0.], [2., 3.]]
        assert_array_equal(obtained, expected)

        look_for = np.array([-1.0, 5.0, 2.1])
        obtained = utils.find_surrounded_as_matrix(look_in, look_for)
        expected = [[-1., -1.], [5., 5.], [2., 3.]]
        assert_array_equal(expected, obtained)

        look_for = np.array([-10, 1.0, 1.1])
        with self.assertRaises(Exception) as context:
            utils.find_surrounded_as_matrix(look_in, look_for)
        self.assertTrue('is out of bound' in str(context.exception))

    def test_calculate_cos_theta(self):
        line_of_sight = np.array([1.0, 1.0, 1.0])
        line_of_sight = line_of_sight / np.linalg.norm(line_of_sight)

        vectors = np.array([[-10, -5, -3.1], [1.0, 1.0, 1.0]])
        vectors = np.array([v / np.linalg.norm(v) for v in vectors])

        expected = [-0.9007,  1.]
        obtained = np.round(utils.calculate_cos_theta(vectors, line_of_sight), 4)
        assert_array_equal(expected, obtained)

        lines_of_sight = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-0.3, 1.0, -20.0]])
        lines_of_sight = np.array([v / np.linalg.norm(v) for v in lines_of_sight])
        obtained = np.round(utils.calculate_cos_theta(vectors, lines_of_sight), 4)
        expected = [[-0.9007, 0.5922, 0.2582],
                    [1., -0.3333, -0.5564]]
        assert_array_equal(expected, obtained)

    def test_calculate_cos_theta_los_x(self):
        vectors = np.array([[1.3, 1.2], [1.1, -0.5]])
        cos_sim = utils.calculate_cos_theta_los_x(vectors)
        assert_array_equal([1.3, 1.1], cos_sim)

        vectors = np.array([[1.3, 1.2, 3.0], [-1.1, -0.5, 1e25]])
        cos_sim = utils.calculate_cos_theta_los_x(vectors)
        assert_array_equal([1.3, -1.1], cos_sim)

    def test_convert_gravity_acceleration_array(self):
        log_g_si = np.array([2.2, 1.3, 2.22])
        log_cgs = log_g_si + 2
        si = up.power(10, log_g_si)
        cgs = up.power(10, log_g_si + 2)

        obtained = utils.convert_gravity_acceleration_array(log_g_si, "log_cgs")
        assert_array_equal(log_cgs, obtained)
        obtained = utils.convert_gravity_acceleration_array(log_g_si, "SI")
        assert_array_equal(si, obtained)
        obtained = utils.convert_gravity_acceleration_array(log_g_si, "cgs")
        assert_array_equal(cgs, obtained)
        obtained = utils.convert_gravity_acceleration_array(log_g_si, "log_SI")
        assert_array_equal(log_g_si, obtained)

    def test_cosine_similarity(self):
        vs_2d_1 = np.array([[0.0, 1.0], [1.0, 0.0]])
        vs_2d_2 = np.array([[0.0, 10.0], [15.0, 0.0]])
        r_1 = utils.cosine_similarity(vs_2d_1[0], vs_2d_1[1])
        r_2 = utils.cosine_similarity(vs_2d_2[0], vs_2d_2[1])

        self.assertEqual(round(r_1, 4), 0)
        self.assertEqual(round(r_2, 4), 0)

        vs_3d = np.array([[1.0, 1.0, 1.33], [-1, 0.0, -3.1]])
        r_3 = utils.cosine_similarity(vs_3d[0], vs_3d[1])

        self.assertEqual(round(r_3, 4), -0.8101)

    def test_is_empty(self):
        empty = [None, dict(), list(), np.array([]), np.nan, pd.NaT]
        result = np.array([utils.is_empty(val) for val in empty])
        self.assertTrue(np.all(result))

        not_empty = [0, 1, -1, dict(x=1), [1], [0, 0], np.array([0])]
        result = np.array([utils.is_empty(val) for val in not_empty])

        self.assertTrue(np.all(up.invert(result)))

    def test_IterableQueue(self):
        q = Queue()
        for i in range(10):
            q.put(i)
        expected = [i for i in range(10)]
        obtained = [val for val in utils.IterableQueue(q)]
        assert_array_equal(expected, obtained)

#
# def check_face_duplicity(faces=None, points=None):
#     """
#     checks if `faces` contains the same faces
#
#     :param faces: np.array of simplices
#     :return:
#     """
#     checklist = [set(xx) for xx in faces]
#     for ii, face1 in enumerate(checklist):
#         for face2 in checklist[ii + 1:]:
#             if face1 == face2:
#                 return False
#
#     return True

    def test_find_surrounded(self):
        look_in = [-1.5, -0.1, 0.0, 0.1, 1.1, 10]
        look_fors = [-10, -1.5, -0.15, 1.2, 10, 21]

        expected = ["raise", [-1.5, -1.5], [-1.5, -0.1], [1.1, 10.0], [10.0, 10.0], "raise"]

        for look_for, expect in zip(look_fors, expected):
            if expect == "raise":
                with self.assertRaises(Exception) as context:
                    utils.find_surrounded(look_in, look_for)
                self.assertTrue('Any value in `look_for` is out of bound of `look_in`' in str(context.exception))
            else:
                obtained = utils.find_surrounded(look_in, look_for)
                self.assertEqual(obtained, expect)

    def test_find_idx_of_nearest(self):
        values = np.array([0.951, 0.851])
        array = np.array([0.81, 0.85, 0.95])

        expected = [2, 1]
        obtained = utils.find_idx_of_nearest(array, values)
        assert_array_equal(expected, obtained)

        array = np.array([0.951, 0.851])
        values = np.array([0.81, 0.85, 0.95])

        expected = [1, 1, 0]
        obtained = utils.find_idx_of_nearest(array, values)
        assert_array_equal(expected, obtained)

    def _test_plane_projection(self, expected, plane):
        points = np.array([[1, 1, 1], [0.3, 0.1, -5], [-2, -3, -4.1]])
        obtained = utils.plane_projection(points, plane, keep_3d=False)
        self.assertTrue(np.all(obtained == expected))

    def test_plane_projection_xy(self):
        expeceted = np.array([[1, 1], [0.3, 0.1], [-2, -3]])
        self._test_plane_projection(expected=expeceted, plane="xy")

    def test_plane_projection_yz(self):
        expeceted = np.array([[1, 1], [0.1, -5], [-3, -4.1]])
        self._test_plane_projection(expected=expeceted, plane="yz")

    def test_plane_projection_zz(self):
        expeceted = np.array([[1, 1], [0.3, -5], [-2, -4.1]])
        self._test_plane_projection(expected=expeceted, plane="zx")

    def test_str_repalce_scalar(self):
        string = 'Hello there, how are you.'
        expected = 'Hello there, how are MIKE.'
        obtained = utils.str_repalce(string, 'you', 'MIKE')
        self.assertEqual(expected, obtained)

    def test_str_repalce_vector(self):
        string = 'Hello there, how are you.'
        expected = 'Hi there, how are MIKE.'
        obtained = utils.str_repalce(string, ['you', 'Hello'], ['MIKE', 'Hi'])
        self.assertEqual(expected, obtained)
