import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../es')

from es import plain_es


def test_quadratic_1d():
    x0 = 5.
    eta = 0.01
    mu = 1.
    sigma = 1.

    def f(x):
        return (x - x0) ** 2

    res = plain_es.optimize(f, eta, mu, sigma, record_history=True)

    assert(res['success'])
    assert(abs(res['mu'] - x0) < 1e-2)
