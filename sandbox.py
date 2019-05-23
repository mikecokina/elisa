import numpy as np
from pypex.poly2d.intersection.linter import multiple_determinants
from copy import copy
import matplotlib.pyplot as plt
from time import time

in_touch = False
tol = 1e-9

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

star_time = time()

m1, n1 = face.shape
m2, n2 = polygon.shape

face_edges = np.empty((m1, 2, 2))
face_edges[:, 0, :] = face
face_edges[:, 1, :] = np.roll(face, axis=0, shift=-1)

polygon_edges = np.empty((m2, 2, 2))
polygon_edges[:, 0, :] = polygon
polygon_edges[:, 1, :] = np.roll(polygon, axis=0, shift=-1)

dface = face_edges[:, 1, :] - face_edges[:, 0, :]
dpolygon = polygon_edges[:, 1, :] - polygon_edges[:, 0, :]

corr_dface = np.repeat(dface, m2, axis=0)
corr_dpolygon = np.tile(dpolygon, (m1, 1))

matrix = np.empty((m1 * m2, n1, n2))
matrix[:, 0, :] = corr_dface
matrix[:, 1, :] = corr_dpolygon

determinants = multiple_determinants(matrix)

intersection_status = np.empty(m1*m2, dtype=np.bool)
intersection_segment = np.empty(m1*m2, dtype=np.bool)
msg = np.chararray(m1*m2, itemsize=9)
intr_ptx = np.empty((m1*m2, 2), dtype=np.float)
distance = np.empty(m1*m2, dtype=np.float)

non_intersections = np.abs(determinants) < tol
if non_intersections.any():
    problem_face = np.repeat(face, m2, axis=0)[non_intersections]
    problem_dface = np.repeat(dface, m2, axis=0)[non_intersections]
    a1, b1 = -problem_dface[:, 1], problem_dface[:, 0]

    face_dface = np.empty((problem_face.shape[0], 2, problem_face.shape[1]), dtype=dface.dtype)
    face_dface[:, 0, :] = problem_face
    face_dface[:, 1, :] = problem_dface
    c1 = multiple_determinants(face_dface)

    problem_polygon_edges = np.tile(polygon_edges, (face.shape[0], 1, 1))[non_intersections]
    c2 = -(a1 * problem_polygon_edges[:, 1, 0] + b1 * problem_polygon_edges[:, 1, 1])
    d = np.abs(c2 - c1) / (np.sqrt(np.power(a1, 2) + np.sqrt(np.power(b1, 2))))

    intersection_status[non_intersections] = False
    intersection_segment[non_intersections] = False
    intr_ptx[non_intersections] = np.NaN
    distance[non_intersections] = d
    msg[non_intersections] = np.str('PARALLEL')

    overlaps = copy(non_intersections)
    overlaps[non_intersections] = d < tol
    intersection_status[overlaps] = True
    # here should be the actual segment but nah...
    intersection_segment[overlaps] = True
    msg[overlaps] = 'OVERLAP'

intersections = ~non_intersections
ok_dface = np.repeat(dface, m2, axis=0)[intersections]
ok_dpolygon = np.tile(dpolygon, (m1, 1))[intersections]
ok_face_edges = np.repeat(face_edges, m2, axis=0)[intersections]
ok_polygon_edges = np.tile(polygon_edges, (m1, 1, 1))[intersections]

p1_p3 = ok_face_edges[:, 0, :] - ok_polygon_edges[:, 0, :]

dp2_p1_p3matrix = np.empty((ok_dface.shape[0], 2, p1_p3.shape[1]), dtype=dface.dtype)
dp2_p1_p3matrix[:, 0, :] = ok_dpolygon
dp2_p1_p3matrix[:, 1, :] = p1_p3

dp1_p1_p3matrix = np.empty((ok_dface.shape[0], 2, p1_p3.shape[1]), dtype=dface.dtype)
dp1_p1_p3matrix[:, 0, :] = ok_dface
dp1_p1_p3matrix[:, 1, :] = p1_p3

d = determinants[intersections]
u = (multiple_determinants(dp2_p1_p3matrix) / d) + 0.0
v = (multiple_determinants(dp1_p1_p3matrix) / d) + 0.0

eval_method = np.less_equal if in_touch else np.less
intr = ok_face_edges[:, 0, :] + (u[:, np.newaxis] * ok_dface)

u_in_range = np.logical_and(eval_method(0.0, u), eval_method(u, 1.0))
v_in_range = np.logical_and(eval_method(0.0, v), eval_method(v, 1.0))

true_intersections = np.logical_and(u_in_range, v_in_range)

intersection_status[intersections] = True
intersection_segment[intersections] = np.logical_and(u_in_range, v_in_range)
intr_ptx[intersections] = np.NaN

intr_candidates = intr_ptx[intersections]
intr_candidates[true_intersections] = intr[true_intersections]
intr_ptx[intersections] = intr_candidates

distance[intersections] = np.NaN

msg[intersections] = np.str('MISSED')
intr_candidates = msg[intersections]
intr_candidates[true_intersections] = 'INTERSECT'
msg[intersections] = intr_candidates

print('Elapsed time: {:.6f}'.format(time()-star_time))
print(msg)
print(intr_ptx)
print(intersection_segment)

# plt.plot(face[0:2, 0], face[0:2, 1], c='blue')
# plt.plot(polygon[0:2, 0], polygon[0:2, 1], c='red')
# plt.scatter(intr_ptx[0, 0], intr_ptx[0, 1], c='black')
plt.plot(face[:, 0], face[:, 1], c='blue')
plt.plot(polygon[:, 0], polygon[:, 1], c='red')
# plt.scatter(intr_ptx[0, 0], intr_ptx[0, 1], c='black')
plt.scatter(intr_ptx[:, 0], intr_ptx[:, 1], c='black')
plt.show()


