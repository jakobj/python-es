import numpy as np
import pytest
import sys

sys.path.append('../')

from es import exponential_natural_es as xnes
import functions

TOLERANCE_1D = 1e-8
TOLERANCE_2D = 1e-7
MAX_ITER = 3000
SEED = np.random.randint(2 ** 32)  # store seed to be able to reproduce errors


def test_quadratic_1d():

    with pytest.raises(ValueError):
        xnes.optimize(lambda: None, np.array([0.]), np.array([0.]))


def test_quadratic_2d():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (x0, y0) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10))):

        def f(x):
            return functions.f_2d(x, x0, y0)

        res = xnes.optimize(f, np.array([mu_x, mu_y]), np.array([[sigma_x, 0.],
                                                                 [0., sigma_y]]), max_iter=MAX_ITER)

        assert(abs(res['mu'][0] - x0) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - y0) < TOLERANCE_2D), SEED


def test_quadratic_2d_non_isotropic():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (x0, y0) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10))):

        def f(x):
            return functions.f_2d_nonisotropic(x, x0, y0)

        res = xnes.optimize(f, np.array([mu_x, mu_y]), np.array([[sigma_x, 0.],
                                                                 [0., sigma_y]]), max_iter=MAX_ITER, record_history=True)

        assert(abs(res['mu'][0] - x0) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - y0) < TOLERANCE_2D), SEED

        # check that width of search distribution in x direction is in
        # most steps smaller than in y direction
        history_cov = np.array(res['history_cov'])
        assert(np.sum([sx < sy for sx, sy in zip(history_cov[:, 0, 0], history_cov[:, 1, 1])])
               >= 0.9 * len(history_cov))


def test_rosenbrock():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 2.

    for (mu_x, mu_y), (a, b) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(0.5, 1.5, 10), np.random.uniform(90., 110., 10))):

        theo_min = [a, a ** 2]

        def f(x):
            return -1. * ((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2)

        def f_xy(x, y):
            return f([x, y])

        res = xnes.optimize(f, np.array([mu_x, mu_y]), np.array([[sigma_x**2, 0.],
                                                                 [0., sigma_y**2]]),
                            learning_rate_mu=None, learning_rate_sigma=None,
                            max_iter=MAX_ITER, record_history=True)

        assert(abs(res['mu'][0] - theo_min[0]) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - theo_min[1]) < TOLERANCE_2D), SEED


def test_correlated_gaussian():
    np.random.seed(SEED)

    mu0_x = 2.
    mu0_y = -1.

    for (mu_x, mu_y), (sigma_x, sigma_y) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(0.5, 1.5, 10), np.random.uniform(0.5, 1.5, 10))):

        def f(x):
            import scipy.stats
            mean = np.array([mu0_x, mu0_y])
            cov = np.array([[.5, -.9],
                            [-.9, 2.]])
            return scipy.stats.multivariate_normal(mean=mean, cov=cov).pdf(x)
        theo_min = [mu0_x, mu0_y]

        def f_xy(x, y):
            return f(np.array([x, y]).T)

        res = xnes.optimize(f, np.array([mu_x, mu_y]), np.array([[sigma_x**2, 0.],
                                                                 [0., sigma_y**2]]),
                            learning_rate_mu=None, learning_rate_sigma=None,
                            max_iter=MAX_ITER, record_history=True)

        assert(abs(res['mu'][0] - theo_min[0]) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - theo_min[1]) < TOLERANCE_2D), SEED
