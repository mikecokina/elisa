import os
from itertools import groupby
from pathlib import Path

import pandas as pd

BASE_PATH = Path(__file__).parent
FILE_PATH = BASE_PATH / "data" / "GaiaDR2_RevisedPassbands.dat"
OUTPUT_DIR = BASE_PATH / "output"

COL_TO_FILE = {
    "gPb": "Gaia.2010.G",
    "bpPb": "Gaia.2010.BP",
    "rpPb": "Gaia.2010.RP"
}


def read_native():
    # Read the file using whitespace as the delimiter
    df = pd.read_csv(FILE_PATH,
                     delim_whitespace=True, header=None,
                     names=["wl", "gPb", "gPbError", "bpPb", "bpPbError", "rpPb", "rpPbError"])
    return df


def find_effective_block(lst):
    blocks = []
    # Enumerate the list to keep track of indices
    for is_less_than_one, group in groupby(enumerate(lst), key=lambda x: 1.0 > x[1] > 1e-6):
        if is_less_than_one:  # Only consider groups where the value is < 1.0
            group_list = list(group)  # List of (index, value) pairs
            start = group_list[0][0]
            end = group_list[-1][0]
            blocks.append((start, end))

    print(blocks)
    return blocks[0]


def main():
    dr2 = read_native()
    wavelength_col = "wl"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for col, filename in COL_TO_FILE.items():
        df = pd.DataFrame(
            {
                "wavelength": dr2[wavelength_col],
                "throughput": dr2[col]
            }
        )

        df["throughput"][df["throughput"] > 1.0] = 0.0
        df["throughput"][df["throughput"] < 0.0] = 0.0
        tp = dr2[col]
        left_index, right_index = find_effective_block(tp)
        df = df.loc[left_index:right_index]

        filepath = OUTPUT_DIR / f"{filename}.csv"
        df.to_csv(filepath, index=False)


if __name__ == '__main__':
    main()

