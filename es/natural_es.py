import numpy as np

from . import lib


def optimize(func, mu, sigma,
             learning_rate_mu=None, learning_rate_sigma=None, population_size=40,
             sigma_lower_bound=1e-12, max_iter=2000,
             fitness_shaping=True, record_history=False):
    """
    Evolutionary strategies using the natural gradient of multinormal search distributions.
    Does not consider covariances between parameters.
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """

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
        z = np.random.normal(mu, sigma, size=(population_size, *np.shape(mu)))
        fitness = np.fromiter((func(zi) for zi in z), np.float)

        if fitness_shaping:
            order, utility = lib.utility(fitness)
            z = z[order]
        else:
            utility = fitness

        # update parameter of search distribution via natural gradient descent
        mu -= learning_rate_mu * 1. / population_size * np.dot(utility, z - mu)
        sigma -= learning_rate_sigma * 1. / population_size * np.dot(utility, (z - mu) ** 2 - sigma ** 2) * 1. / (2. * sigma)

        # enforce lower bound on sigma to avoid negative values
        if np.any(sigma < sigma_lower_bound):
            sigma[sigma < sigma_lower_bound] = sigma_lower_bound

        if record_history:
            history_mu.append(mu.copy())
            history_sigma.append(sigma.copy())
            history_pop.append(z.copy())

        generation += 1

        # exit if max iterations reached
        if generation > max_iter:
            break

    return {'mu': mu, 'sigma': sigma, 'history_mu': history_mu, 'history_sigma': history_sigma, 'history_pop': history_pop}
