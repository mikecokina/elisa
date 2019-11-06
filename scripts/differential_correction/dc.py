import os.path as op
import numpy as np
import json

# q = 3
# T = 1000
# fn = lambda x: q * x + T

__N__ = 20
__PHASES__ = np.arange(0, __N__, 1)

with open(op.join(op.dirname(__file__), "data", "lc.json"), "r") as f:
    __OBSERVED__ = np.array(list(json.loads(f.read()).values()))

dq, dT = 0.2, 10
q0, T0 = 1, 900
q_factor, T_factor = 0.1, 0.1


def synthetic(qq, TT, phases):
    return qq * phases + TT


def next_step(q, T):
    # x_n+1 = x_n + g Derror(x_n)

    diff_q, diff_T = derivatives(q, T)

    qn = q + (q_factor * diff_q)
    Tn = T + (T_factor * diff_T)

    return qn, Tn


def derivatives(qn, Tn):
    dq_plus, dq_minus = qn + (dq / 2.0), qn - (dq / 2.0)
    dT_plus, dT_minus = Tn + (dT / 2.0), Tn - (dT / 2.0)

    diff_q = (error_fn(dq_plus, Tn) - error_fn(dq_minus, Tn)) / dq
    diff_T = (error_fn(qn, dT_plus) - error_fn(qn, dT_minus)) / dT
    
    return -diff_q, -diff_T


def error_fn(q, T):
    return np.sum(np.power(__OBSERVED__ - synthetic(q, T, __PHASES__), 2) / __OBSERVED__)


def r_squared(q, T):
    # Coefficient of determination
    observed_mean = np.mean(__OBSERVED__)

    variability = np.sum(np.power(__OBSERVED__ - observed_mean, 2))
    residual = np.sum(np.power(__OBSERVED__ - synthetic(q, T, __PHASES__), 2))
    return 1 - (residual / variability)


def main():
    print("q = 3, T = 1000")
    qn, Tn = q0, T0

    for i in range(10000):
        qn, Tn = next_step(qn, Tn)
        print(f"step {i}, qn: {qn}, Tn: {Tn}, \tr_squared: {r_squared(qn, Tn)}, \terror: {error_fn(qn, Tn)}")


main()
