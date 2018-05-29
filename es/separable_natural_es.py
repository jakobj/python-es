import multiprocessing as mp
import numpy as np
import torch

from . import lib


def optimize(func, mu, sigma,
             learning_rate_mu=None, learning_rate_sigma=None, population_size=None,
             max_iter=2000,
             fitness_shaping=True, mirrored_sampling=True, record_history=False,
             rng=None,
             parallel_threads=None,
             optimizer=torch.optim.SGD):
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

    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, int):
        rng = np.random.RandomState(seed=rng)

    generation = 0
    history_mu = []
    history_sigma = []
    history_pop = []
    history_fitness = []

    # convert mu to torch Variable and construct optimizer; force
    # torch to use double representation
    mu_torch = torch.autograd.Variable(torch.DoubleTensor(mu.copy()), requires_grad=True)
    optimizer_mu = optimizer([mu_torch], lr=learning_rate_mu)

    while True:

        # use numpy representation for generating individuals
        mu_numpy = mu_torch.detach().numpy()

        s = rng.normal(0, 1, size=(population_size, *np.shape(mu_numpy)))
        z = mu_numpy + sigma * s

        if mirrored_sampling:
            z = np.vstack([z, mu_numpy - sigma * s])
            s = np.vstack([s, -s])

        if parallel_threads is None:
            fitness = np.fromiter((func(zi) for zi in z), np.float)
        else:
            pool = mp.Pool(processes=parallel_threads)
            fitness = np.fromiter(pool.map(func, z), np.float)
            pool.close()
            pool.join()

        ni = np.logical_not(np.isnan(fitness))
        z = z[ni]
        s = s[ni]
        fitness = fitness[ni]

        if fitness_shaping:
            order, utility = lib.utility(fitness)
            s = s[order]
            z = z[order]
        else:
            utility = fitness

        if record_history:
            history_mu.append(mu_numpy.copy())
            history_sigma.append(sigma.copy())
            history_pop.append(z.copy())
            history_fitness.append(fitness.copy())

        # exit if max iterations reached
        if generation > max_iter or np.all(sigma < 1e-10):
            break

        # update parameters of search distribution via natural
        # gradient descent in natural coordinates

        # set gradient and use optimizer to update mu
        mu_torch.grad = torch.autograd.Variable(torch.DoubleTensor(-sigma * np.dot(utility, s)))
        optimizer_mu.step()

        # manually update sigma
        sigma *= np.exp(learning_rate_sigma / 2. * np.dot(utility, s ** 2 - 1))

        generation += 1

    return {'mu': mu_numpy,
            'sigma': sigma,
            'history_mu': history_mu,
            'history_sigma': history_sigma,
            'history_fitness': history_fitness,
            'history_pop': history_pop}
