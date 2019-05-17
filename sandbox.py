import itertools
import pandas as pd
import numpy as np

from elisa.conf.config import ATM_MODEL_DATAFRAME_WAVE, ATM_MODEL_DATAFRAME_FLUX
from elisa.engine import atm


def find_atm_defined_wavelength(atm_containers):
    for atm_container in atm_containers:
        if atm_container is not None:
            return atm_container.model[ATM_MODEL_DATAFRAME_WAVE]


def remap_passbanded_unique_atms_to_matrix(atm_containers, fpaths_map):
    total = max(list(itertools.chain.from_iterable(fpaths_map.values()))) + 1
    wavelengths_defined = find_atm_defined_wavelength(atm_containers)
    wavelengths_length = len(wavelengths_defined)
    models_matrix = np.zeros(total * wavelengths_length).reshape(total, wavelengths_length)

    for atm_container in atm_containers:
        if atm_container is not None:
            models_matrix[fpaths_map[atm_container.fpath]] = atm_container.model[ATM_MODEL_DATAFRAME_FLUX]
    return models_matrix


# total = max(list(itertools.chain.from_iterable(fpaths_map.values()))) + 1
#     models_arr = np.array([None] * total)
#     for model in models:
#         if model is not None:
#             models_arr[fpaths_map[model.fpath]] = model
#     return models_arr


class Atm(object):
    def __init__(self, df, path):
        self.model = df
        self.fpath = path


df_10 = pd.DataFrame({ATM_MODEL_DATAFRAME_WAVE: [1, 2, 3, 4], ATM_MODEL_DATAFRAME_FLUX: [10, 20, 30, 40]})
df_9 = pd.DataFrame({ATM_MODEL_DATAFRAME_WAVE: [1, 2, 3, 4], ATM_MODEL_DATAFRAME_FLUX: [9, 19, 29, 38]})
df_8 = pd.DataFrame({ATM_MODEL_DATAFRAME_WAVE: [1, 2, 3, 4], ATM_MODEL_DATAFRAME_FLUX: [5, 15, 22, 7]})

containers = [Atm(c, p) for p, c in zip(['10000', '9000', '8000'], [df_10, df_9, df_8])]
fpaths = ['10000', '9000', '9000', '9000', None, '8000']
_, fmap = atm.unique_atm_fpaths(fpaths)

matrix = remap_passbanded_unique_atms_to_matrix(containers, fmap)
print(matrix)
