import numpy as np
import matplotlib.path as mpltpath

from scipy.spatial.qhull import ConvexHull
from elisa.engine import const
from elisa.engine.binary_system import geo


def compute_circular_synchronous_lightcurve(self, **kwargs):
    # get orbital motion from kwargs
    # get eclipses
    # prepare rotated (orb motion, inclination) points and normals
    # apply darkside filter !!!!!!! just indices information !!!!!!!

    # NOTE: extention of system primary and secondary points due to horizont faces fractalisation
    # apply eclipse filter !!!!!!! just indices information !!!!!!!

    orbital_motion = kwargs.pop("positions")
    eclipses = geo.get_eclipse_boundaries(self, 1.0)
    # system_positions_container = self.prepare_system_positions_container(orbital_motion=orbital_motion)
    # system_positions_container = system_positions_container.darkside_filter()

    # todo: it makes more sense to do eclipse filter in same way as darkside filter
    # todo: rewrite it in the future

    # for container in system_positions_container:
    #     pass

    import pickle
    container = pickle.load(open("container.pickle", "rb"))

    counter_part = {"primary": "secondary", "secondary": "primary"}
    cover_component = 'secondary' if 0.0 < container.position.azimut < const.PI else 'primary'
    cover_object = getattr(container, cover_component)
    undercover_object = getattr(container, counter_part[cover_component])
    undercover_visible_point_indices = list(set(undercover_object.faces[undercover_object.indices].flatten()))

    cover_object_obs_visible_projection = geo.plane_projection(
        cover_object.points[
            list(set(cover_object.faces[cover_object.indices].flatten()))
        ], "yz"
    )

    undercover_object_obs_visible_projection = geo.plane_projection(
        undercover_object.points[
            list(set(undercover_object.faces[undercover_object.indices].flatten()))
        ], "yz"
    )

    cover_bound = ConvexHull(cover_object_obs_visible_projection)
    hull_points = cover_object_obs_visible_projection[cover_bound.vertices]
    bb_path = mpltpath.Path(hull_points)

    out_of_bound = np.invert(bb_path.contains_points(undercover_object_obs_visible_projection))
    undercover_visible_point_indices = np.array(undercover_visible_point_indices)[out_of_bound]

    undercover_faces = np.array([const.FALSE_FACE_PLACEHOLDER] * int(len(undercover_object.normals)))
    undercover_faces[undercover_object.indices] = undercover_object.faces[undercover_object.indices]

    eclipse_faces_visibility = np.isin(undercover_faces, undercover_visible_point_indices)

    full_visible = np.array([np.all(face) for face in eclipse_faces_visibility])
    invisible = np.array([np.all(face) for face in np.invert(eclipse_faces_visibility)])
    partial_visible = np.invert(full_visible | invisible)

    # coverage = np.zeros(len(undercover_object.faces))
    # coverage[full_visible] = 1.0
    # coverage[invisible] = 0.0
    # # todo: overage for partial visible
    # coverage[partial_visible] = -1

    # from matplotlib import pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # p = undercover_object.points[
    #     undercover_visible_point_indices
    # ]
    # xs, ys, zs = p.T[0], p.T[1], p.T[2]
    # ax.scatter(xs, ys, zs, c="b", marker="o", s=0.1)
    # plt.show()


    # from matplotlib import pyplot as plt
    # x, y = undercover_object_visible_projection.T[0][out_of_bound], undercover_object_visible_projection.T[1][out_of_bound]
    # plt.scatter(x, y, marker="o", c="b", s=0.1)
    # plt.xlabel("y")
    # plt.xlabel("z")
    # plt.axis("equal")
    #
    # plt.show()

    from matplotlib import pyplot as plt
    # points = np.concatenate((container._primary.points, container._secondary.points), axis=0)
    # faces = np.concatenate((container._primary.faces, container._secondary.faces + len(container._primary.points)), axis=0)
    # indices = np.concatenate((container._primary.indices, container._secondary.indices + len(container._primary.normals)), axis=0)
    # faces = faces[indices]
    #
    idx = partial_visible
    points = container.primary.points
    faces = container.primary.faces[idx]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    clr = 'b'

    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)

    ax.view_init(0, 0)

    plot = ax.plot_trisurf(
        points.T[0], points.T[1],
        points.T[2], triangles=faces,
        antialiased=True, shade=False, color=clr)
    plot.set_edgecolor('black')
    plt.show()




    exit()
    # import os
    # for idx, pos in enumerate(system_positions_container):
    #     p = pos.primary.points
    #     s = pos.secondary.points
    #
    #     pi = pos.primary.indices
    #     si = pos.secondary.indices
    #
    #     p = geo.plane_projection(p, plane="yz")
    #     s = geo.plane_projection(s, plane="yz")
    #     c1, c2 = ["b"] * len(list(set(self.primary.faces[pi].flatten()))), \
    #              ["r"] * len(list(set(self.secondary.faces[si].flatten())))
    #     c = c1 + c2
    #     ps = np.concatenate((p[list(set(self.primary.faces[pi].flatten()))],
    #                          s[list(set(self.secondary.faces[si].flatten()))]), axis=0)
    #
    #     path = os.path.join("C:\\Users\\d59637\\Documents\\tmp", "{}.png".format(idx))
    #     geo.to_png(x=ps.T[0], y=ps.T[1], x_label="y", y_label="z", c=c, fpath=path)
    #
    # exit()

    # print(eclipses)
    # exit()
    #
    # orbital_motion = kwargs.pop("positions")
    # print(orbital_motion)
    #
    # import pickle
    #
    # # pickle.dump(self.primary.faces, open("facesp.pickle", "wb"))
    # # pickle.dump(self.primary.points, open("pointsp.pickle", "wb"))
    # # pickle.dump(self.primary.normals, open("normalsp.pickle", "wb"))
    # #
    # # pickle.dump(self.secondary.faces, open("facess.pickle", "wb"))
    # # pickle.dump(self.secondary.points, open("pointss.pickle", "wb"))
    # # pickle.dump(self.secondary.normals, open("normalss.pickle", "wb"))
    #
    # facesp = pickle.load(open("facesp.pickle", "rb"))
    # pointsp = pickle.load(open("pointsp.pickle", "rb"))
    # normalsp = pickle.load(open("normalsp.pickle", "rb"))
    #
    # facess = pickle.load(open("facess.pickle", "rb"))
    # pointss = pickle.load(open("pointss.pickle", "rb"))
    # normalss = pickle.load(open("normalss.pickle", "rb"))
    #
    # visible_p = geo.darkside_filter(sight_of_view=BINARY_SIGHT_OF_VIEW, normals=normalsp)
    # visible_s = geo.darkside_filter(sight_of_view=BINARY_SIGHT_OF_VIEW, normals=normalss)
    #
    # # visible = np.concatenate((visible_p, visible_s + len(normalsp)), axis=0)
    # # faces = np.concatenate((facesp, facess + len(pointsp)), axis=0)
    # # points = np.concatenate((pointsp, pointss), axis=0)























    # faces = faces[visible]
    # points = points
    #
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure(figsize=(7, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect('equal')
    #
    # clr = 'b'
    # pts = points
    # fcs = faces
    #
    # plot = ax.plot_trisurf(
    #     pts[:, 0], pts[:, 1],
    #     pts[:, 2], triangles=fcs,
    #     antialiased=True, shade=False, color=clr)
    #
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.set_zlim(-1, 1)
    # ax.view_init(0, -np.degrees(0.09424778))
    #
    # plot.set_edgecolor('black')
    #
    # plt.show()
    #
    #
    # pass
    # geo.darkside_filter()

    # compute on filtered atmospheres (doesn't meeter how will be filtered)
    # primary_radiance = \
    #     atm.NaiveInterpolatedAtm.radiance(_temperature, _logg, self.primary.metallicity, config.ATM_ATLAS, **kwargs)

    # primary_radiance = \
    #     atm.NearestAtm.radiance(_temperature, _logg, self.primary.metallicity, config.ATM_ATLAS, **kwargs)


if __name__ == "__main__":
    pass
