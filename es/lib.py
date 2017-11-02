import numpy as np


def utility(fitness):
    n = len(fitness)
    order = np.argsort(fitness)[::-1]
    fitness = fitness[order]

    utility = [np.max([0, np.log((n / 2) + 1)]) - np.log(k + 1) for k in range(n)]
    utility = utility / np.sum(utility) - 1. / n

    return order, utility
