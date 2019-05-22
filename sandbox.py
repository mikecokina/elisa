import numpy as np


face = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.6, 0.7],
        [-0.2, 0.3]
    ])

polygon = np.array(
    [
        [0.4, 0.2],
        [0.4, -0.5],
        [1.5, -0.5],
        [1.5, 0.3],
        [0.75, 0.3]
    ]
)

face_edges = np.zeros((len(face), 2, 2))
face_edges[:, 0, :] = face
face_edges[:, 1, :] = np.roll(face, axis=0, shift=1)

polygon_edges = np.zeros((len(polygon), 2, 2))
polygon_edges[:, 0, :] = polygon
polygon_edges[:, 1, :] = np.roll(polygon, axis=0, shift=1)

dp1 = face_edges[:, 0] - face_edges[:, 1]
dp2 = polygon_edges[:, 0] - polygon_edges[:, 1]


m1, n1 = dp1.shape
m2, n2 = dp2.shape

matrix = np.zeros((m1, m2, n1+n2), dtype=dp1.dtype)
matrix[:, :, :n1] = dp1[:, None, :]
matrix[:, :, n1:] = dp2
matrix = matrix.reshape(-1, 2, 2)

determinatns = matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]





