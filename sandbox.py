import itertools
import pandas as pd
import numpy as np

from elisa.conf.config import ATM_MODEL_DATAFRAME_WAVE, ATM_MODEL_DATAFRAME_FLUX
from elisa.engine import atm


# total = max(list(itertools.chain.from_iterable(fpaths_map.values()))) + 1
#     models_arr = np.array([None] * total)
#     for model in models:
#         if model is not None:
#             models_arr[fpaths_map[model.fpath]] = model
#     return models_arr


class Atm(object):
    def __init__(self, df, path, t):
        self.model = df
        self.fpath = path
        self.temperature = t


df_10 = pd.DataFrame({ATM_MODEL_DATAFRAME_WAVE: [1, 2, 3, 4], ATM_MODEL_DATAFRAME_FLUX: [10, 20, 30, 40]})
df_9 = pd.DataFrame({ATM_MODEL_DATAFRAME_WAVE: [1, 2, 3, 4], ATM_MODEL_DATAFRAME_FLUX: [9, 19, 29, 38]})
df_8 = pd.DataFrame({ATM_MODEL_DATAFRAME_WAVE: [1, 2, 3, 4], ATM_MODEL_DATAFRAME_FLUX: [5, 15, 22, 7]})

containers = [Atm(c, p, int(p)) for p, c in zip(['10000', '9000', '8000'], [df_10, df_9, df_8])]
temperatures = [9501, 9000, 8345]

top_atm = [containers[0], containers[1], containers[1]]
bottom_atm = [containers[1], None, containers[2]]
weights = atm.NaiveInterpolatedAtm.compute_interpolation_weights(temperatures, top_atm, bottom_atm)


fpaths = ['10000', '9000', '9000', '9000', None, '8000']
_, fmap = atm.unique_atm_fpaths(fpaths)

matrix = atm.remap_passbanded_unique_atm_to_matrix(containers, fmap)
top_matrix, bottom_matrix = matrix[:len(matrix) // 2], matrix[len(matrix) // 2:]

interpolated = (weights * (top_matrix.T - bottom_matrix.T) + bottom_matrix.T).T

print(top_matrix)
print(interpolated)
print(bottom_matrix)
