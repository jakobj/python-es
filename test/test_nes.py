import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../')

from es import natural_es as nes
import functions

TOLERANCE_1D = 1e-9
TOLERANCE_2D = 1e-9
TOLERANCE_ROSENBROCK = 1e-2
MAX_ITER = 3000
MAX_ITER_ROSENBROCK = 100000
SEED = np.random.randint(2 ** 32)  # store seed to be able to reproduce errors


def test_quadratic_1d():
    np.random.seed(SEED)

    sigma = 1.

    for mu, x0 in zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)):

        def f(x):
            return functions.f_1d(x, x0)

        res = nes.optimize(f, np.array([mu]), np.array([sigma]), max_iter=MAX_ITER, record_history=True)

        assert(abs(res['mu'] - x0) < TOLERANCE_1D), SEED


def test_quadratic_2d():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (x0, y0) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10))):

        def f(x):
            return functions.f_2d(x, x0, y0)

        res = nes.optimize(f, np.array([mu_x, mu_y]), np.array([sigma_x, sigma_y]), max_iter=MAX_ITER)

        assert(abs(res['mu'][0] - x0) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - y0) < TOLERANCE_2D), SEED


def test_rosenbrock():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (a, b) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(0.5, 1.5, 10), np.random.uniform(90., 110., 10))):

        theo_min = [a, a ** 2]

        def f(x):
            return functions.f_rosenbrock(x, a, b)

        res = nes.optimize(f, np.array([mu_x, mu_y]), np.array([sigma_x, sigma_y]),
                           # learning_rate_mu=0.2, learning_rate_sigma=0.001,
                           max_iter=MAX_ITER_ROSENBROCK)

        assert(abs(res['mu'][0] - theo_min[0]) < TOLERANCE_ROSENBROCK), SEED
        assert(abs(res['mu'][1] - theo_min[1]) < TOLERANCE_ROSENBROCK), SEED
