import os.path as op
import pandas as pd

DATA = op.join(op.abspath(op.dirname(__file__)), "data")
OUTPUT = op.join(op.abspath(op.dirname(__file__)), "claret20_wd.out")

BAND_MAP = {
    "Ke": "Kepler",
    "Tes": "TESS",
    "G": "GaiaDR2",
    "u'": "SLOAN.SDSS.u",
    "g'": "SLOAN.SDSS.g",
    "r'": "SLOAN.SDSS.r",
    "i'": "SLOAN.SDSS.i",
    "z'": "SLOAN.SDSS.z",
    "U": "Generic.Bessell.U",
    "B": "Generic.Bessell.B",
    "V": "Generic.Bessell.V",
    "R": "Generic.Bessell.R",
    "I": "Generic.Bessell.I"
}


LAW_MAP = {
    "linear": "lin",
    "logarithmic": "log",
    "square_root": "sqrt"
}


def get_data(fname, ld):
    cols = {
        "linear": ["D-DA-3D", "band", "logg", "Teff", "ZR", "u(Ke)", "Kesig", "I(Ke)"],
        "logarithmic": ["D-DA-3D", "band", "logg", "Teff", "ZR", "e(Ke)", "f(Ke)", "Kesig", "I(Ke)"],
        "square_root": ["D-DA-3D", "band", "logg", "Teff", "ZR", "c(Ke)", "d(Ke)", "Kesig", "I(Ke)"],
    }
    col_map = {
        "linear": {
            "logg": "gravity",
            "Teff": "temperature",
            "u(Ke)": "xlin",
            "ZR": "Z",
            "band": "band"
        },
        "logarithmic": {
            "logg": "gravity",
            "Teff": "temperature",
            "e(Ke)": "xlog",
            "f(Ke)": "ylog",
            "ZR": "Z",
            "band": "band"
        },
        "square_root": {
            "logg": "gravity",
            "Teff": "temperature",
            "c(Ke)": "xsqrt",
            "d(Ke)": "ysqrt",
            "ZR": "Z",
            "band": "band"
        }
    }

    col_filter = {
        "linear": ['gravity', 'temperature', 'Z', 'xlin'],
        "logarithmic": ['gravity', 'temperature', 'Z', 'xlog', 'ylog'],
        "square_root": ['gravity', 'temperature', 'Z', 'xsqrt', 'ysqrt']
    }

    fpath = op.join(DATA, fname)
    df = pd.read_csv(fpath, sep="\s+", comment="#", header=None)
    df.columns = cols[ld]
    df = df.rename(columns=col_map[ld])
    df = df[col_filter[ld]].astype(float)
    return df


def get_filename(law, band, metallicity):
    pass


def main():
    fname = ["tableu.dat", "tableef.dat", "tablecd.dat"]
    ld = ["linear", "logarithmic", "square_root"]

    for _ld, _fname in zip(ld, fname):
        df = get_data(_fname, ld=_ld)

    pass


if __name__ == '__main__':
    main()
