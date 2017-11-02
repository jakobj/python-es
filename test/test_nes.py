import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../es')

from es import natural_es as nes

TOLERANCE_1D = 0.05
TOLERANCE_2D = 0.05
SEED = np.random.randint(2 ** 32)  # store seed to be able to reproduce errors


def test_quadratic_1d():
    np.random.seed(SEED)

    learning_rate_mu = 0.1
    learning_rate_sigma = 0.05
    sigma = 1.

    for mu, x0 in zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)):

        def f(x):
            return (x - x0) ** 2

        res = nes.optimize(f, np.array(mu), np.array(sigma), learning_rate_mu, learning_rate_sigma, max_iter=5000)

        assert(abs(res['mu'] - x0) < TOLERANCE_1D), SEED


def test_quadratic_2d():
    np.random.seed(SEED)

    learning_rate_mu = 0.1
    learning_rate_sigma = 0.05
    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (x0, y0) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10))):

        def f(x):
            return (x[0] - x0) ** 2 + (x[1] - y0) ** 2

        res = nes.optimize(f, np.array([mu_x, mu_y]), np.array([sigma_x, sigma_y]), learning_rate_mu, learning_rate_sigma, max_iter=5000)

        assert(abs(res['mu'][0] - x0) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - y0) < TOLERANCE_2D), SEED
