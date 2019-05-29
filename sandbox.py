from elisa.engine import utils
import numpy as np

from elisa.engine.utils import find_surrounded_as_matrix


fin = np.array([1, 2, 3, 4, 5])
ffor = np.array([3.4, 10])

find_surrounded_as_matrix(fin, ffor)
