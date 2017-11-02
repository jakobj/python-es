import numpy as np
import sys

sys.path.append('../es')

from es import plain_es as pes

TOLERANCE_1D = 0.15
TOLERANCE_2D = 0.15
SEED = np.random.randint(2 ** 32)  # store seed to be able to reproduce errors


def test_quadratic_1d():
    np.random.seed(SEED)

    eta = 0.01
    sigma = 1.

    for mu, x0 in zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)):

        def f(x):
            return (x - x0) ** 2

        res = pes.optimize(f, eta, np.array(mu), np.array(sigma), max_iter=3000)

        assert(abs(res['mu'] - x0) < TOLERANCE_1D), SEED


def test_quadratic_2d():
    np.random.seed(SEED)

    eta = 0.01
    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (x0, y0) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10))):

        def f(x):
            return (x[0] - x0) ** 2 + (x[1] - y0) ** 2

        res = pes.optimize(f, eta, np.array([mu_x, mu_y]), np.array([sigma_x, sigma_y]), max_iter=4000)

        assert(abs(res['mu'][0] - x0) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - y0) < TOLERANCE_2D), SEED
