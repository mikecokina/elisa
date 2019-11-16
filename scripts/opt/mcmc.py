import time

import emcee
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = f"{int(os.cpu_count())}"


def random_sign():
    random = np.random.randint(0, 2)
    return 1 if random else -1


def model(x0, x):
    a, b = x0
    return (a * x) + b


def ln_likelihood(xn, *args):
    xs, ys, yerr = args
    synthetic = model(xn, xs)

    return -0.5 * np.sum(np.power((synthetic - ys) / yerr, 2))


def ln_prior(xn):
    bound = ((0, 5), (0, 5))
    in_bounds = [bound[idx][0] <= xn[idx] <= bound[idx][1] for idx in range(0, len(bound))]
    return 0.0 if np.all(in_bounds) else -np.inf


def ln_prob(xn, *args):
    xs, ys, yerrs = args
    lp = ln_prior(xn)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(xn, xs, ys, yerrs)  # recall if lp not -inf, its 0, so this just returns likelihood


def eval_mcmc(p0, nwalkers, niter, ndim, _lnprob, *args):
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=_lnprob, args=args)
    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state


def main():
    np.random.seed(int(time.time()))
    nwalkers = 500
    x = (1.3, 3.21)
    x0 = (1.1, 4.0)
    ndim, niter = len(x0), 1024

    # p0 = np.array([np.array(x0) + 1e-2 * np.random.randn(ndim) for _ in range(nwalkers)])
    p0 = np.random.uniform(1.0, 5.0, (nwalkers, ndim))

    xs = np.arange(0, 10, 0.01)
    ys = model(x, xs)

    # signs = np.array([random_sign() for _ in range(len(ys))])
    # yerrs = ys + (ys * 0.05 * signs)
    yerr = np.mean(ys) * 0.02

    args = (xs, ys, yerr)
    sampler, pos, prob, state = eval_mcmc(p0, nwalkers, niter, ndim, ln_prob, *args)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    flat_samples = sampler.get_chain(flat=True)
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)

        print(f"{mcmc[1]}, +{q[1]} / -{q[0]}")


if __name__ == "__main__":
    main()
