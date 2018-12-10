import unittest
import numpy as np
from engine import utils
from numpy import testing


def test_cartesian_to_spherical():
    input1 = np.array([[1., 0., 0.], [0., 0., 1.]])
    input2 = [1, 0, 0]

    expected_output1 = np.array([[1, 0, np.pi/2], [1, 0, 0]])
    expected_output2 = np.array([1, 0, np.pi/2])
    expected_output3 = np.array([1, 0, 90])

    output1 = utils.cartesian_to_spherical(input1)
    output2 = utils.cartesian_to_spherical(input2)
    output3 = utils.cartesian_to_spherical(input2, degrees=True)

    testing.assert_array_equal(output1, expected_output1)
    testing.assert_array_equal(output2, expected_output2)
    testing.assert_array_equal(output3, expected_output3)


def test_spherical_to_cartesian():
    input1 = np.array([[1, 0, np.pi / 2], [1, 0, 0]])
    input2 = np.array([1, 0, np.pi / 2])

    expected_output1 = np.array([[1., 0., 0.], [0., 0., 1.]])
    expected_output2 = [1, 0, 0]

    output1 = utils.spherical_to_cartesian(input1)
    output2 = utils.spherical_to_cartesian(input2)

    testing.assert_array_almost_equal(output1, expected_output1)
    testing.assert_array_almost_equal(output2, expected_output2)


def check_face_duplicity(faces=None, points=None):
    """
    checks if `faces` contains the same faces

    :param faces: np.array of simplices
    :return:
    """
    checklist = [set(xx) for xx in faces]
    for ii, face1 in enumerate(checklist):
        for face2 in checklist[ii + 1:]:
            if face1 == face2:
                return False

    return True
