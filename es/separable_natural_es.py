import glob
import multiprocessing as mp
import numpy as np
import pickle
from . import lib


def load_checkpoint():
    filenames = glob.glob('checkpoint-*.pkl')
    if filenames:
        most_recent_checkpoint = max(filenames, key=lambda x: int(x.split('-')[1].split('.')[0]))
        with open(most_recent_checkpoint, 'rb') as f:
            most_recent_state = pickle.load(f)
        return most_recent_state
    else:
        return None


def create_checkpoint(locals_dict, whitelist, label):
    state = extract_keys_from_dict(locals_dict, whitelist)
    with open('checkpoint-{}.pkl'.format(label), 'wb') as f:
        pickle.dump(state, f)


def extract_keys_from_dict(d, whitelist, blacklist=[]):
    return {key: value for key, value in d.items() if key in whitelist and key not in blacklist}


def optimize(func, mu, sigma,
             learning_rate_mu=None, learning_rate_sigma=None, population_size=None,
             max_iter=2000,
             fitness_shaping=True, mirrored_sampling=True, record_history=False,
             rng=None,
             parallel_threads=None,
             checkpoint_interval=None,
             load_existing_checkpoint=False):
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

    mu = mu.copy()
    sigma = sigma.copy()
    generation = 0
    history_mu = []
    history_sigma = []
    history_pop = []
    history_fitness = []

    mutable_locals = ['rng', 'mu', 'sigma', 'generation', 'history_mu', 'history_sigma', 'history_pop', 'history_fitness']

    if load_existing_checkpoint:
        state = load_checkpoint()
        if state:
            rng = state['rng']
            mu = state['mu']
            sigma = state['sigma']
            generation = state['generation'] + 1
            history_mu = state['history_mu']
            history_sigma = state['history_sigma']
            history_pop = state['history_pop']
            history_fitness = state['history_fitness']

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

        if record_history:
            history_mu.append(mu.copy())
            history_sigma.append(sigma.copy())
            history_pop.append(z.copy())
            history_fitness.append(fitness.copy())

        if checkpoint_interval is not None and generation % checkpoint_interval == 0:
            create_checkpoint(locals(), mutable_locals, generation)

        generation += 1

        # exit if max iterations reached
        if generation > max_iter or np.all(sigma < 1e-10):
            break

    return lib.create_results_dict(mu, sigma, history_mu, history_sigma, history_fitness, history_pop)
