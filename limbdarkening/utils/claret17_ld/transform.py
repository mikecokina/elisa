import os.path as op
import pandas as pd
from elisa.ld import get_ld_table_filename

DATA = op.join(op.abspath(op.dirname(__file__)), "data")
OUTPUT = op.join(op.abspath(op.dirname(__file__)), "claret17.tess")


def get_data(fname, ld):
    col_map = {
        "linear": {
            "logg": "gravity",
            "Teff": "temperature",
            "uLSM": "xlin"
        },
        "logarithmic": {
            "logg": "gravity",
            "Teff": "temperature",
            "eLSM": "xlog",
            "fLSM": "ylog"
        },
        "square_root": {
            "logg": "gravity",
            "Teff": "temperature",
            "cLSM": "xsqrt",
            "dLSM": "ysqrt"
        }
    }

    col_filter = {
        "linear": ['gravity', 'temperature', 'Z', 'xlin'],
        "logarithmic": ['gravity', 'temperature', 'Z', 'xlog', 'ylog'],
        "square_root": ['gravity', 'temperature', 'Z', 'xsqrt', 'ysqrt']
    }

    fpath = op.join(DATA, fname)
    df = pd.read_csv(fpath, sep=";", comment="#")
    df = df.rename(columns=col_map[ld])
    df = df[col_filter[ld]].astype(float)
    return df


def main():

    fname = ["tess_lin.csv", "tess_log.csv", "tess_sqrt.csv"]
    ld = ["linear", "logarithmic", "square_root"]

    for _ld, _fname in zip(ld, fname):
        df = get_data(_fname, ld=_ld)
        df = df.sort_values(by=["Z", "temperature", "gravity"])
        gb = df.groupby('Z')
        for x in gb.groups:
            filename = get_ld_table_filename("TESS", x, law=_ld)
            _df = gb.get_group(x)
            _df = _df.loc[:, _df.columns != 'Z']
            _df.to_csv(op.join(OUTPUT, filename), index=False)


if __name__ == "__main__":
    main()
