import numpy as np

from . import lib


def optimize(func, learning_rate, mu, sigma,
             sigma_lower_bound=0.1, max_iter=2000, population_size=30,
             tol=0.01, fitness_shaping=True, record_history=False):
    """
    Evolutionary strategies using the plain gradient of multinormal search distributions.
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """

    generation = 0
    history_mu = []
    history_sigma = []
    history_pop = []
    success = False

    while True:
        z = np.random.normal(mu, sigma, size=population_size)
        fitness = np.array([func(zi) for zi in z])

        if fitness_shaping:
            order, utility = lib.utility(fitness)
            z = z[order]
        else:
            utility = fitness

        # update parameter of search distribution via plain gradient descent
        mu -= learning_rate * 1. / population_size * np.dot(z - mu, utility) * 1. / sigma ** 2
        sigma -= learning_rate * 1. / population_size * np.dot((z - mu) ** 2 - sigma ** 2, utility) * 1. / sigma ** 3

        # enforce lower bound on sigma to avoid instabilities
        if sigma <= sigma_lower_bound:
            sigma = sigma_lower_bound

        if record_history:
            history_mu.append(mu)
            history_sigma.append(sigma)
            history_pop.append(z)

        generation += 1

        # exit if converged or max iterations reached
        if len(history_mu) > 100 and np.std(history_mu[-100:]) < tol * np.abs(np.mean(history_mu[-100:])):
            success = True
            break
        elif generation > max_iter:
            success = False
            break

    return {'mu': mu, 'sigma': sigma, 'success': success, 'history_mu': history_mu, 'history_sigma': history_sigma, 'history_pop': history_pop}
