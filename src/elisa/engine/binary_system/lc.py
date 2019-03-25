import numpy as np

from elisa.engine.binary_system import geo
from elisa.engine.const import BINARY_SIGHT_OF_VIEW


def compute_circular_synchronous_lightcurve(self, **kwargs):
    # get orbital motion from kwargs
    # get eclipses
    # prepare rotated (orb motion, inclination) points and normals
    # apply darkside filter !!!!!!! just indices information !!!!!!!

    # NOTE: extention of system primary and secondary points due to horizont faces fractalisation
    # apply eclipse filter !!!!!!! just indices information !!!!!!!

    orbital_motion = kwargs.pop("positions")
    eclipses = geo.get_eclipse_boundaries(self, 1.0)

    # orbital_motion = [
    #     [0, 1.0, np.radians(90), np.radians(90), 0.0],
    #     [1, 1.0, np.radians(135), np.radians(135), 0.0],
    #     [2, 1.0, np.radians(180), np.radians(180), 0.0]
    # ]
    system_positions_container = self.prepare_system_positions_container(orbital_motion=orbital_motion)

    import os
    for idx, pos in enumerate(system_positions_container):
        p = pos.primary.points
        s = pos.secondary.points

        p = geo.plane_projection(p, plane="yz")
        s = geo.plane_projection(s, plane="yz")
        c1, c2 = ["b"] * len(p), ["r"] * len(s)
        c = c1 + c2
        ps = np.concatenate((p, s), axis=0)

        path = os.path.join("C:\\Users\\d59637\\Documents\\tmp", "{}.png".format(idx))
        geo.to_png(x=ps.T[0], y=ps.T[1], x_label="y", y_label="z", c=c, fpath=path)

    exit()

    print(eclipses)
    exit()

    orbital_motion = kwargs.pop("positions")
    print(orbital_motion)

    import pickle

    # pickle.dump(self.primary.faces, open("facesp.pickle", "wb"))
    # pickle.dump(self.primary.points, open("pointsp.pickle", "wb"))
    # pickle.dump(self.primary.normals, open("normalsp.pickle", "wb"))
    #
    # pickle.dump(self.secondary.faces, open("facess.pickle", "wb"))
    # pickle.dump(self.secondary.points, open("pointss.pickle", "wb"))
    # pickle.dump(self.secondary.normals, open("normalss.pickle", "wb"))

    facesp = pickle.load(open("facesp.pickle", "rb"))
    pointsp = pickle.load(open("pointsp.pickle", "rb"))
    normalsp = pickle.load(open("normalsp.pickle", "rb"))

    facess = pickle.load(open("facess.pickle", "rb"))
    pointss = pickle.load(open("pointss.pickle", "rb"))
    normalss = pickle.load(open("normalss.pickle", "rb"))

    visible_p = geo.darkside_filter(sight_of_view=BINARY_SIGHT_OF_VIEW, normals=normalsp)
    visible_s = geo.darkside_filter(sight_of_view=BINARY_SIGHT_OF_VIEW, normals=normalss)

    # visible = np.concatenate((visible_p, visible_s + len(normalsp)), axis=0)
    # faces = np.concatenate((facesp, facess + len(pointsp)), axis=0)
    # points = np.concatenate((pointsp, pointss), axis=0)























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
