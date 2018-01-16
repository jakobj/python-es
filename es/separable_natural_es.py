import numpy as np

from . import lib


def optimize(func, mu, sigma,
             learning_rate_mu=None, learning_rate_sigma=None, population_size=None,
             max_iter=2000,
             fitness_shaping=True, mirrored_sampling=True, record_history=False):
    """
    Evolution strategies using the natural gradient of multinormal search distributions in natural coordinates.
    Does not consider covariances between parameters.
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """

    if not isinstance(mu, np.ndarray):
        raise TypeError('mu needs to be of type np.ndarray')
    if not isinstance(sigma, np.ndarray):
        raise TypeError('sigma needs to be of type np.ndarray')

    if learning_rate_mu is None:
        learning_rate_mu = lib.default_learning_rate_mu()
    if learning_rate_sigma is None:
        learning_rate_sigma = lib.default_learning_rate_sigma(mu.size)
    if population_size is None:
        population_size = lib.default_population_size(mu.size)

    generation = 0
    history_mu = []
    history_sigma = []
    history_pop = []

    while True:
        s = np.random.normal(0, 1, size=(population_size, *np.shape(mu)))
        z = mu + sigma * s

        if mirrored_sampling:
            z = np.vstack([z, mu - sigma * s])
            s = np.vstack([s, -s])

        fitness = np.fromiter((func(zi) for zi in z), np.float)

        if fitness_shaping:
            order, utility = lib.utility(fitness)
            s = s[order]
            z = z[order]
        else:
            utility = fitness

        # update parameter of search distribution via natural gradient descent in natural coordinates
        mu -= learning_rate_mu * sigma * np.dot(utility, s)
        sigma *= np.exp(-learning_rate_sigma / 2. * np.dot(utility, s ** 2 - 1))

        if record_history:
            history_mu.append(mu.copy())
            history_sigma.append(sigma.copy())
            history_pop.append(z.copy())

        generation += 1

        # exit if max iterations reached
        if generation > max_iter or np.all(sigma < 1e-10):
            break

    return {'mu': mu, 'sigma': sigma, 'history_mu': history_mu, 'history_sigma': history_sigma, 'history_pop': history_pop}
