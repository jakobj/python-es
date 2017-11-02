import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../es')

from es import plain_es as pes

TOLERANCE_1D = 0.15
TOLERANCE_2D = 0.15
MAX_ITER = 3000
SEED = np.random.randint(2 ** 32)  # store seed to be able to reproduce errors


def test_quadratic_1d():
    np.random.seed(SEED)

    sigma = 1.

    for mu, x0 in zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)):

        def f(x):
            return (x - x0) ** 2

        res = pes.optimize(f, np.array(mu), np.array(sigma), population_size=50, max_iter=MAX_ITER, record_history=False)

        # plt.plot(res['history_mu'], 'b')
        # plt.plot(res['history_sigma'], 'm')
        # plt.axhline(x0, color='k')
        # plt.show()

        assert(abs(res['mu'] - x0) < TOLERANCE_1D), SEED


def test_quadratic_2d():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (x0, y0) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10))):

        def f(x):
            return (x[0] - x0) ** 2 + (x[1] - y0) ** 2

        res = pes.optimize(f, np.array([mu_x, mu_y]), np.array([sigma_x, sigma_y]), population_size=50, max_iter=MAX_ITER)

        assert(abs(res['mu'][0] - x0) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - y0) < TOLERANCE_2D), SEED
