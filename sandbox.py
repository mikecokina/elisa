from elisa.engine import utils
import numpy as np

min = 0.8
max = 2.8
num = 21

phase = np.linspace(min, max, num)

print(utils.base_phase_interval(phase))
