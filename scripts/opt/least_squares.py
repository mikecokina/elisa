import numpy as np
from scipy.optimize import least_squares

observed_1 = np.array([-9.21573874e-04, 1.01448840e-02, 2.82313822e-02, 7.09549163e-02,
                       1.31072574e-01, 2.00072999e-01, 2.89786826e-01, 3.95786150e-01,
                       5.14238924e-01, 6.54729483e-01, 8.06915307e-01, 9.70304177e-01,
                       1.15886022e+00, 1.35868605e+00, 1.56800731e+00, 1.80093299e+00,
                       2.05437980e+00, 2.31992794e+00, 2.59869849e+00, 2.89542346e+00])

observed_2 = np.array([3.20907843, 3.16211988, 3.10613138, 3.05872992, 3.01267257,
                       2.959448, 2.91088683, 2.86256115, 2.81063892, 2.76470448,
                       2.71441531, 2.65927918, 2.61326022, 2.56246105, 2.50510731,
                       2.45530799, 2.4099798, 2.36070294, 2.30859849, 2.25839846])


def fn1(a, x):
    return a * np.power(x, 2)


def fn2(a, x):
    return a - x


def model(x0, *args):
    xs, ys1, ys2 = args
    # return fn1(x0, xs) - ys1
    # return fn2(x0, xs) - ys2
    # return (fn1(x0, xs) - ys1) + (fn2(x0, xs) - ys2)
    return np.array([np.sum(np.power(fn1(x0, xs) - ys1, 2)), np.sum(np.power(fn2(x0, xs) - ys2, 2))])


def fit(xs, ys1, ys2):
    x0 = 3.0
    args = (xs, ys1, ys2)
    result = least_squares(model, x0, bounds=(0, 5), args=args, xtol=1e-15)
    print(result.x)


def prepare_data():
    a = 3.21
    xs = np.arange(0, 1, 0.05)
    bias = (np.random.rand(len(xs)) - 0.5) / 1e2
    print(repr(fn1(a, xs) + bias))
    print(repr(fn2(a, xs) + bias))


def main():
    # prepare_data()
    xs = np.arange(0, 1, 0.05)
    fit(xs, ys1=observed_1, ys2=observed_2)


if __name__ == "__main__":
    main()
