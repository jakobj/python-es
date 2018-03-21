import numpy as np


def default_population_size(dimensions):
    """
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """
    return 4 + int(np.floor(3 * np.log(dimensions)))


def default_learning_rate_mu():
    """
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """
    return 1


def default_learning_rate_sigma(dimensions):
    """
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """
    return (3 + np.log(dimensions)) / (5. * np.sqrt(dimensions))


def default_learning_rate_sigma_exponential(dimensions):
    """
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """
    return (9 + 3. * np.log(dimensions)) / (5. * dimensions * np.sqrt(dimensions))


def default_learning_rate_B_exponential(dimensions):
    """
    Learning rate for B, seems to be much too large when using default value from Wierstra et al. (2014).
    Hence reduce significantly.
    """
    return 0.005 * (9 + 3. * np.log(dimensions)) / (5. * dimensions * np.sqrt(dimensions))


def utility(fitness):
    """
    Utility function for fitness shaping.
    See Wierstra et al. (2014). Natural evolution strategies. Journal of Machine Learning Research, 15(1), 949-980.
    """

    n = len(fitness)
    order = np.argsort(fitness)[::-1]
    fitness = fitness[order]

    utility = [np.max([0, np.log((n / 2) + 1)]) - np.log(k + 1) for k in range(n)]
    utility = utility / np.sum(utility) - 1. / n

    return order, utility
