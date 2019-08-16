from elisa.conf.config import PASSBAND_TABLES, PASSBAND_DATAFRAME_THROUGHPUT, PASSBAND_DATAFRAME_WAVE
from os.path import join as pjoin
import pandas as pd


with open(pjoin(PASSBAND_TABLES, "GaiaDR2"), "r") as f:
    fl = list()
    x, y = [], []
    df = pd.DataFrame(columns=[PASSBAND_DATAFRAME_WAVE, PASSBAND_DATAFRAME_THROUGHPUT])
    while True:

        try:
            line = str(f.readline()).strip()
            d = line.split(" ")
            d[0]; d[2]

            x.append(d[0])
            y.append(d[2])

        except IndexError:
            try:
                float(d[0])
            except:
                break

            continue

    df[PASSBAND_DATAFRAME_THROUGHPUT] = y
    df[PASSBAND_DATAFRAME_WAVE] = x

df.to_csv(pjoin(PASSBAND_TABLES, "GaiaDR2.csv"), index=False)



            # fl.append([line.split(" ")[0], line.split(" ")[1]])
        # except:
        #     continue

