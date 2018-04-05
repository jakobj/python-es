import numpy as np
import scipy.linalg

from . import lib


def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def optimize(func, mu, cov,
             learning_rate_mu=None, learning_rate_sigma=None, learning_rate_B=None,
             population_size=None, max_iter=2000,
             fitness_shaping=True, mirrored_sampling=True, record_history=False):
    """
    Evolution strategies using the natural gradient of multinormal search distributions in natural coordinates.
    Does not consider covariances between parameters.
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """

    if len(mu) == 1:
        raise ValueError('Not possible for one-dimensional optimization. Use separable natural es instead.')

    if not isinstance(mu, np.ndarray):
        raise TypeError('mu needs to be of type np.ndarray')
    if not isinstance(cov, np.ndarray):
        raise TypeError('sigma needs to be of type np.ndarray')

    if learning_rate_mu is None:
        learning_rate_mu = lib.default_learning_rate_mu()
    if learning_rate_sigma is None:
        learning_rate_sigma = lib.default_learning_rate_sigma_exponential(mu.size)
    if learning_rate_B is None:
        learning_rate_B = lib.default_learning_rate_B_exponential(mu.size)
    if population_size is None:
        population_size = lib.default_population_size(mu.size)
    if not is_positive_definite(cov):
        raise ValueError('covariance matrix needs to be positive semidefinite')

    generation = 0
    history_mu = []
    history_cov = []
    history_pop = []

    # Find Cholesky decomposition of covariance matrix
    A = np.linalg.cholesky(cov).T
    assert(np.sum(np.abs(np.dot(A.T, A) - cov)) < 1e-12), 'Chochelsky decomposition failed'

    # Decompose A into scalar step and normalized covariance factor B
    sigma = (abs(np.linalg.det(A)))**(1. / mu.size)
    B = A / sigma

    if record_history:
        history_mu.append(mu.copy())
        cov = sigma**2 * np.dot(B.T, B)
        history_cov.append(cov.copy())
        history_pop.append(np.empty((population_size, *np.shape(mu))))

    while True:
        assert(abs(np.linalg.det(B) - 1.) < 1e-12), 'determinant of root of covariance matrix unequal one'

        s = np.random.normal(0, 1, size=(population_size, *np.shape(mu)))
        z = mu + sigma * np.dot(s, B)

        if mirrored_sampling:
            z = np.vstack([z, mu - sigma * np.dot(s, B)])
            s = np.vstack([s, -s])

        fitness = np.fromiter((func(zi) for zi in z), np.float)

        if fitness_shaping:
            order, utility = lib.utility(fitness)
            s = s[order]
            z = z[order]
        else:
            utility = fitness

        grad_J_d = np.dot(s.T, utility)
        grad_J_M = np.dot(
            np.array([(np.outer(s[i], s[i]) - np.eye(len(mu))) for i in range(len(s))]).T,
            utility)
        grad_J_sigma = np.trace(grad_J_M) / len(mu)
        grad_J_B = grad_J_M - grad_J_sigma * np.eye(len(mu))

        # update parameter of search distribution via natural gradient descent in natural coordinates
        mu += learning_rate_mu * sigma * np.dot(B, grad_J_d)
        sigma *= np.exp(learning_rate_sigma / 2. * grad_J_sigma)
        B = np.dot(B, scipy.linalg.expm(learning_rate_B / 2. * grad_J_B))

        if record_history:
            history_mu.append(mu.copy())
            cov = sigma ** 2 * np.dot(B.T, B)
            history_cov.append(cov.copy())
            history_pop.append(z.copy())

        generation += 1

        # exit if max iterations reached
        if generation > max_iter or sigma ** 2 < 1e-20:
            break

    return {'mu': mu, 'sigma': sigma, 'history_mu': history_mu, 'history_cov': history_cov, 'history_pop': history_pop}
