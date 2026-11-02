import numpy as np
from numpy.testing import assert_allclose
from elisa.numba_functions.reflection_effect import gamma_primary, gamma_secondary

from elisa.numba_functions.operations import (
    create_distance_vector_matrix,
    calculate_lengths_in_3d_array,
    divide_points_in_array_by_constants,
)


def test_gamma_primary_basic():
    """Test gamma_primary with small arrays."""
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    join_vector = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    ])
    # gamma_primary[ii, jj] = normals[ii] . join_vector[ii, jj]
    # [0, 0]: [1,0,0] . [1,0,0] = 1.0
    # [0, 1]: [1,0,0] . [0,1,0] = 0.0
    # [1, 0]: [0,1,0] . [0,1,0] = 1.0
    # [1, 1]: [0,1,0] . [1,0,0] = 0.0
    expected = np.array([[1.0, 0.0], [1.0, 0.0]])
    result = gamma_primary(normals, join_vector)
    assert_allclose(result, expected)


def test_gamma_secondary_basic():
    """Test gamma_secondary with small arrays."""
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    join_vector = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
    ])
    # gamma_secondary[ii, jj] = -normals[jj] . join_vector[ii, jj]
    # [0, 0]: -[1,0,0] . [1,0,0] = -1.0
    # [0, 1]: -[0,1,0] . [0,1,0] = -1.0
    # [1, 0]: -[1,0,0] . [0,1,0] = -0.0
    # [1, 1]: -[0,1,0] . [1,0,0] = -0.0
    expected = np.array([[-1.0, -1.0], [-0.0, -0.0]])
    result = gamma_secondary(normals, join_vector)
    assert_allclose(result, expected)


def test_create_distance_vector_matrix_basic():
    """Test create_distance_vector_matrix with simple known inputs.

    :returns: None
    :rtype: None
    """
    points1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    result = create_distance_vector_matrix(points1, points2)

    # Expected: result[i, j] = points2[j] - points1[i]
    # [0, 0]: [1, 0, 0] - [0, 0, 0] = [1, 0, 0]
    # [0, 1]: [0, 1, 0] - [0, 0, 0] = [0, 1, 0]
    # [1, 0]: [1, 0, 0] - [1, 0, 0] = [0, 0, 0]
    # [1, 1]: [0, 1, 0] - [1, 0, 0] = [-1, 1, 0]
    expected = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]],
    ])

    assert_allclose(result, expected)
    assert result.shape == (2, 2, 3)


def test_create_distance_vector_matrix_shape():
    """Test create_distance_vector_matrix output shape is correct.

    :returns: None
    :rtype: None
    """
    points1 = np.random.rand(5, 3)
    points2 = np.random.rand(7, 3)

    result = create_distance_vector_matrix(points1, points2)

    assert result.shape == (5, 7, 3)


def test_calculate_lengths_in_3d_array_basic():
    """Test calculate_lengths_in_3d_array with known vectors.

    :returns: None
    :rtype: None
    """
    matrix = np.array([
        [[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 5.0], [2.0, 0.0, 0.0]],
    ])

    result = calculate_lengths_in_3d_array(matrix)

    # Expected lengths:
    # [0, 0]: sqrt(3^2 + 4^2 + 0^2) = sqrt(9 + 16) = 5.0
    # [0, 1]: sqrt(1^2 + 0^2 + 0^2) = 1.0
    # [1, 0]: sqrt(0^2 + 0^2 + 5^2) = 5.0
    # [1, 1]: sqrt(2^2 + 0^2 + 0^2) = 2.0
    expected = np.array([[5.0, 1.0], [5.0, 2.0]])

    assert_allclose(result, expected)
    assert result.shape == (2, 2)


def test_calculate_lengths_in_3d_array_unit_vectors():
    """Test calculate_lengths_in_3d_array with unit vectors."""
    matrix = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    ])

    result = calculate_lengths_in_3d_array(matrix)

    expected = np.array([
        [1.0, 1.0],
        [1.0, np.sqrt(2.0)],
    ])

    assert_allclose(result, expected)


def test_divide_points_in_array_by_constants_basic():
    """Test divide_points_in_array_by_constants with known values."""
    matrix = np.array([
        [[2.0, 4.0, 6.0], [10.0, 20.0, 30.0]],
        [[4.0, 8.0, 12.0], [5.0, 10.0, 15.0]],
    ])

    coefficients = np.array([
        [2.0, 10.0],
        [4.0, 5.0],
    ])

    result = divide_points_in_array_by_constants(matrix, coefficients)

    # Expected: result[i, j, k] = matrix[i, j, k] / coefficients[i, j]
    # [0, 0]: [2/2, 4/2, 6/2] = [1, 2, 3]
    # [0, 1]: [10/10, 20/10, 30/10] = [1, 2, 3]
    # [1, 0]: [4/4, 8/4, 12/4] = [1, 2, 3]
    # [1, 1]: [5/5, 10/5, 15/5] = [1, 2, 3]
    expected = np.array([
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
    ])

    assert_allclose(result, expected)
    assert result.shape == matrix.shape


def test_divide_points_in_array_by_constants_normalization():
    """Test divide_points_in_array_by_constants for vector normalization."""
    # Create vectors with known lengths
    matrix = np.array([
        [[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 5.0], [2.0, 0.0, 0.0]],
    ])

    # Use lengths as coefficients to normalize
    coefficients = np.array([
        [5.0, 1.0],  # lengths of vectors in matrix
        [5.0, 2.0],
    ])

    result = divide_points_in_array_by_constants(matrix, coefficients)

    # After division by length, all vectors should have length 1
    result_lengths = calculate_lengths_in_3d_array(result)
    expected_lengths = np.ones((2, 2))

    assert_allclose(result_lengths, expected_lengths)


def test_divide_points_in_array_by_constants_dtype():
    """Test divide_points_in_array_by_constants returns float64 dtype."""
    matrix = np.array([
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ])
    coefficients = np.array([[2.0, 2.0]])

    result = divide_points_in_array_by_constants(matrix, coefficients)

    assert result.dtype == np.float64
