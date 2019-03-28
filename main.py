import pickle
import numpy as np

import matplotlib.path as mpltpath
from matplotlib import pyplot as plt
from scipy.spatial.qhull import ConvexHull
from elisa.engine import const, utils
from elisa.engine.binary_system import geo
from elisa.engine.binary_system.geo import EasyObject
from mpl_toolkits.mplot3d import axes3d


_c = axes3d
pc, sc = "b", "r"


if False:
    theta = 5
    facesp = pickle.load(open("facesp.pickle", "rb"))
    pointsp = pickle.load(open("pointsp.pickle", "rb"))
    normalsp = pickle.load(open("normalsp.pickle", "rb"))

    facess = pickle.load(open("facess.pickle", "rb"))
    pointss = pickle.load(open("pointss.pickle", "rb"))
    normalss = pickle.load(open("normalss.pickle", "rb"))

    primary_reference_point = utils.axis_rotation(theta, np.array([0., 0., 0.]), axis="z", degrees=True)
    secondary_reference_point = utils.axis_rotation(theta, np.array([1., 0., 0.]), axis="z", degrees=True)

    pointsp = utils.axis_rotation(theta, pointsp, axis="z", degrees=True)
    normalsp = utils.axis_rotation(theta, normalsp, axis="z", degrees=True)

    pointss = utils.axis_rotation(theta, pointss, axis="z", degrees=True)
    normalss = utils.axis_rotation(theta, normalss, axis="z", degrees=True)

    primary = EasyObject(pointsp, normalsp, None, facesp)
    secondary = EasyObject(pointss, normalss, None, facess)

    primary.indices = geo.darkside_filter(const.BINARY_SIGHT_OF_VIEW, primary.normals)
    secondary.indices = geo.darkside_filter(const.BINARY_SIGHT_OF_VIEW, secondary.normals)

    # points = np.concatenate((primary.points, secondary.points), axis=0)
    # faces = np.concatenate((primary.faces, secondary.faces + len(primary.points)), axis=0)
    # indices = np.concatenate((primary.indices, secondary.indices + len(primary.normals)), axis=0)
    # faces = faces[indices]
    #
    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    # clr = 'b'
    #
    # ax.set_xlim3d(-2, 2)
    # ax.set_ylim3d(-2, 2)
    # ax.set_zlim3d(-2, 2)
    #
    # ax.view_init(0, 0)
    #
    # plot = ax.plot_trisurf(
    #     points.T[0], points.T[1],
    #     points.T[2], triangles=faces,
    #     antialiased=True, shade=False, color=clr)
    # plot.set_edgecolor('black')
    # plt.show()

    primary_visible_projection = geo.plane_projection(
        primary.points[
            list(set(primary.faces[primary.indices].flatten()))
        ], "yz"
    )

    secondary_visible_projection = geo.plane_projection(
        secondary.points[
            list(set(secondary.faces[secondary.indices].flatten()))
        ], "yz"
    )

    primary_projection = geo.plane_projection(primary.points, "yz")
    secondary_projection = geo.plane_projection(secondary.points, "yz")

    bound = ConvexHull(secondary_visible_projection)
    hull_points = secondary_visible_projection[bound.vertices]
    bb_path = mpltpath.Path(hull_points)

    out_of_bound = [idx for idx, point in enumerate(primary_projection) if not bb_path.contains_points([point])[0]]
    primary_projection = primary_projection[out_of_bound]

    x, y = hull_points.T[0], hull_points.T[1]
    plt.scatter(x, y, marker="o", c=sc, s=0.1)
    plt.xlabel("y")
    plt.xlabel("z")
    plt.axis("equal")

    x, y = primary_projection.T[0], primary_projection.T[1]
    plt.scatter(x, y, marker="o", c=pc, s=0.1)
    plt.xlabel("y")
    plt.xlabel("z")
    plt.axis("equal")

    plt.show()

    # x, y = hull_points.T[0], hull_points.T[1]
    # plt.triplot(x, y, primary.faces[primary.indices], lw=0.2, color=(0, 0, 1, 1))
    # plt.xlabel("y")
    # plt.xlabel("z")
    # plt.axis("equal")
    # plt.xlim(-lim, lim)
    # plt.ylim(-lim, lim)
    # plt.show()




    # x, y = primary_visible_projection.T[0], primary_visible_projection.T[1]
    # plt.scatter(x, y, marker="o", c=pc, s=0.1)
    #
    # x, y = secondary_visible_projection.T[0], secondary_visible_projection.T[1]
    # plt.scatter(x, y, marker="o", c=sc, s=0.1)
    #
    # plt.xlabel("y")
    # plt.xlabel("z")
    # plt.axis("equal")
    # plt.xlim(-lim, lim)
    # plt.ylim(-lim, lim)
    # plt.show()



    # points = np.concatenate((primary.points, secondary.points), axis=0)
    # faces = np.concatenate((primary.faces, secondary.faces + len(primary.points)), axis=0)
    # indices = np.concatenate((primary.indices, secondary.indices + len(primary.normals)), axis=0)
    # faces = faces[indices]
    #
    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    # clr = 'b'
    #
    # ax.set_xlim3d(-2, 2)
    # ax.set_ylim3d(-2, 2)
    # ax.set_zlim3d(-2, 2)
    #
    # ax.view_init(0, 0)
    #
    # plot = ax.plot_trisurf(
    #     points.T[0], points.T[1],
    #     points.T[2], triangles=faces,
    #     antialiased=True, shade=False, color=clr)
    # plot.set_edgecolor('black')
    # plt.show()

    # ax = plt.scatter(x, y, marker="o", c=c, s=1)
    # plt.xlabel("y")
    # plt.xlabel("z")
    # plt.axis("equal")
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    # plt.show()


if True:
    theta = 55
    facesp = pickle.load(open("facesp_over.pickle", "rb"))
    pointsp = pickle.load(open("pointsp_over.pickle", "rb"))
    normalsp = pickle.load(open("normalsp_over.pickle", "rb"))

    facess = pickle.load(open("facess_over.pickle", "rb"))
    pointss = pickle.load(open("pointss_over.pickle", "rb"))
    normalss = pickle.load(open("normalss_over.pickle", "rb"))

    primary_reference_point = utils.axis_rotation(theta, np.array([0., 0., 0.]), axis="z", degrees=True)
    secondary_reference_point = utils.axis_rotation(theta, np.array([1., 0., 0.]), axis="z", degrees=True)

    pointsp = utils.axis_rotation(theta, pointsp, axis="z", degrees=True)
    normalsp = utils.axis_rotation(theta, normalsp, axis="z", degrees=True)

    pointss = utils.axis_rotation(theta, pointss, axis="z", degrees=True)
    normalss = utils.axis_rotation(theta, normalss, axis="z", degrees=True)

    primary = EasyObject(pointsp, normalsp, None, facesp)
    secondary = EasyObject(pointss, normalss, None, facess)

    primary.indices = geo.darkside_filter(const.BINARY_SIGHT_OF_VIEW, primary.normals)
    secondary.indices = geo.darkside_filter(const.BINARY_SIGHT_OF_VIEW, secondary.normals)

    primary_projection = geo.plane_projection(primary.points, "yz")
    secondary_projection = geo.plane_projection(secondary.points, "yz")

    primary_visible_projection = geo.plane_projection(
        primary.points[
            list(set(primary.faces[primary.indices].flatten()))
        ], "yz"
    )

    secondary_visible_projection = geo.plane_projection(
        secondary.points[
            list(set(secondary.faces[secondary.indices].flatten()))
        ], "yz"
    )


    # primary_visible_pindices = list(set(primary.faces[primary.indices].flatten()))
    # secondary_visible_pindices = list(set(secondary[secondary.indices].flatten()))
    #
    # bound = ConvexHull(primary_projection[primary_visible_pindices])
    # hull_points = primary_projection[primary_visible_pindices][bound.vertices]
    # # bb_path = mpltpath.Path(hull_points)
    #
    # # out_of_bound = [idx for idx, point in enumerate(secondary_projection) if not bb_path.contains_points([point])[0]]
    # # secondary_projection = secondary_projection[out_of_bound]
    #
    # lim = .5
    # x, y = primary_projection[primary_visible_pindices].T[0], primary_projection[primary_visible_pindices].T[1]
    # plt.scatter(x, y, marker="o", c=pc, s=0.1)
    # plt.xlabel("y")
    # plt.xlabel("z")
    # plt.axis("equal")
    # plt.xlim(-lim, lim)
    # plt.ylim(-lim, lim)
    #
    # x, y = secondary_projection[secondary_visible_pindices].T[0], secondary_projection[secondary_visible_pindices].T[1]
    # plt.scatter(x, y, marker="o", c=pc, s=0.1)
    # plt.xlabel("y")
    # plt.xlabel("z")
    # plt.axis("equal")
    # plt.xlim(-lim, lim)
    # plt.ylim(-lim, lim)
    #
    # plt.show()

    # points = np.concatenate((primary.points, secondary.points), axis=0)
    # faces = np.concatenate((primary.faces, secondary.faces + len(primary.points)), axis=0)
    # indices = np.concatenate((primary.indices, secondary.indices + len(primary.normals)), axis=0)
    # faces = faces[indices]
    #
    # points = primary.points
    # faces = primary.faces[primary.indices]
    #
    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    # clr = 'b'
    #
    # ax.set_xlim3d(-2, 2)
    # ax.set_ylim3d(-2, 2)
    # ax.set_zlim3d(-2, 2)
    #
    # ax.view_init(0, 0)
    #
    # plot = ax.plot_trisurf(
    #     points.T[0], points.T[1],
    #     points.T[2], triangles=faces,
    #     antialiased=True, shade=False, color=clr)
    # plot.set_edgecolor('black')
    # plt.show()

