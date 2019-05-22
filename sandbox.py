import numpy as np
from elisa.engine import utils, const


look_in = np.array(const.CK_GRAVITY_LIST_ATM)

look_for = np.array([0.0, 0.1, 0.3, 4.0, 5.0])

# v = utils.find_nearest_value_as_matrix(look_in, look_for)
v = utils.find_surrounded_as_matrix(look_in, look_for)

print(const.CK_GRAVITY_LIST_ATM)
print(v)

# x = np.array([0.5, 0.1, 0.2, 0.3, 0.5])
# z = np.zeros(len(x))
# y = np.array([0.5, 0.2])
# duplicate = np.isin(x, y)
#
#
#
# print(duplicate)
#








