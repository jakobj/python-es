import logging
import multiprocessing as mp
import numpy as np
from . import lib

logger = logging.getLogger(__name__)


def optimize(func, mu, sigma,
             learning_rate_mu=None, learning_rate_sigma=None, population_size=None,
             max_iter=2000,
             fitness_shaping=True, mirrored_sampling=True, record_history=False,
             rng=None,
             parallel_threads=None,
             verbosity=logging.WARNING):
    """
    Evolution strategies using the natural gradient of multinormal search distributions in natural coordinates.
    Does not consider covariances between parameters.
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """

    logger_ch = logging.StreamHandler()
    logger_fh = logging.FileHandler('snes.log', 'w')
    logger_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(name)s %(message)s'))
    logger_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(name)s %(message)s'))
    logger.setLevel(verbosity)
    logger.addHandler(logger_ch)
    logger.addHandler(logger_fh)

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

    logger.info('starting evolution with {} individuals per generation on {} threads'.format(population_size * (1 + int(mirrored_sampling)), parallel_threads))
    while True:
        s = rng.normal(0, 1, size=(population_size, *np.shape(mu)))
        z = mu + sigma * s

        if mirrored_sampling:
            z = np.vstack([z, mu - sigma * s])
            s = np.vstack([s, -s])

        generations_list = [generation] * len(z)
        individual_list = range(len(z))
        if parallel_threads is None:
            fitness = np.fromiter((func(zi, gi, ii) for zi, gi, ii in zip(z, generations_list, individual_list)), np.float)
        else:
            with mp.Pool(processes=parallel_threads) as pool:
                fitness = np.fromiter(pool.starmap(func, zip(z, generations_list, individual_list)), np.float)

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

        # update parameter of search distribution via natural gradient descent in natural coordinates
        mu += learning_rate_mu * sigma * np.dot(utility, s)
        sigma *= np.exp(learning_rate_sigma / 2. * np.dot(utility, s ** 2 - 1))

        logger.info('generation {}, average fitness {}'.format(generation, np.mean(fitness)))
        logger.debug('fitness {}'.format(fitness))
        logger.debug('mu {}'.format(mu))
        logger.debug('sigma {}'.format(sigma))

        if record_history:
            history_mu.append(mu.copy())
            history_sigma.append(sigma.copy())
            history_pop.append(z.copy())
            history_fitness.append(fitness.copy())

        generation += 1

        # exit if max iterations reached
        if generation > max_iter:
            logger.info('maximum number of iterations reached - exiting')
            break
        elif np.all(sigma < 1e-10):
            break

    return {'mu': mu,
            'sigma': sigma,
            'history_mu': history_mu,
            'history_sigma': history_sigma,
            'history_fitness': history_fitness,
            'history_pop': history_pop}
