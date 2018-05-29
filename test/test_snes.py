import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

sys.path.append('../')

from es import separable_natural_es as snes
import functions

TOLERANCE_1D = 1e-9
TOLERANCE_2D = 1e-9
TOLERANCE_ROSENBROCK = 1e-2
MAX_ITER = 2000
MAX_ITER_ROSENBROCK = 50000
SEED = np.random.randint(2 ** 32)  # store seed to be able to reproduce errors


def test_quadratic_1d():
    np.random.seed(SEED)

    sigma = 1.

    for mu, x0 in zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)):

        def f(x):
            return functions.f_1d(x, x0)

        res = snes.optimize(f, np.array([mu]), np.array([sigma]),
                            max_iter=MAX_ITER, rng=SEED)

        assert(abs(res['mu'] - x0) < TOLERANCE_1D), SEED


def test_quadratic_2d():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (x0, y0) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10))):

        def f(x):
            return functions.f_2d(x, x0, y0)

        res = snes.optimize(f, np.array([mu_x, mu_y]),
                            np.array([sigma_x, sigma_y]), max_iter=MAX_ITER, rng=SEED)

        assert(abs(res['mu'][0] - x0) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - y0) < TOLERANCE_2D), SEED


def test_quadratic_2d_non_isotropic():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (x0, y0) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10))):

        def f(x):
            return functions.f_2d_nonisotropic(x, x0, y0)

        res = snes.optimize(f, np.array([mu_x, mu_y]),
                            np.array([sigma_x, sigma_y]), max_iter=MAX_ITER,
                            record_history=True, rng=SEED)

        assert(abs(res['mu'][0] - x0) < TOLERANCE_2D), SEED
        assert(abs(res['mu'][1] - y0) < TOLERANCE_2D), SEED

        # check that width of search distribution in x direction is in
        # most steps smaller than in y direction
        history_sigma = np.array(res['history_sigma'])
        assert(np.sum([sx < sy for sx, sy in zip(history_sigma[:, 0], history_sigma[:, 1])])
               >= 0.9 * len(history_sigma))


def test_rosenbrock():
    np.random.seed(SEED)

    sigma_x = 1.
    sigma_y = 1.

    for (mu_x, mu_y), (a, b) in zip(
            zip(np.random.uniform(-5., 5., 10), np.random.uniform(-5., 5., 10)),
            zip(np.random.uniform(0.5, 1.5, 10), np.random.uniform(90., 110., 10))):

        theo_min = [a, a ** 2]

        def f(x):
            return functions.f_rosenbrock(x, a, b)

        res = snes.optimize(f, np.array([mu_x, mu_y]), np.array([sigma_x, sigma_y]),
                            # learning_rate_mu=0.1, learning_rate_sigma=0.00025,
                            max_iter=MAX_ITER_ROSENBROCK,
                            rng=SEED)

        assert(abs(res['mu'][0] - theo_min[0]) < TOLERANCE_ROSENBROCK), SEED
        assert(abs(res['mu'][1] - theo_min[1]) < TOLERANCE_ROSENBROCK), SEED


def test_ann():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    criterion = torch.nn.MSELoss()
    trials = 2
    inner_iterations = 50
    in_features = 2
    hidden_units = 4
    out_features = 1

    class ANN(torch.nn.Module):

        def __init__(self):
            super().__init__()

            self.fc1 = torch.nn.Linear(in_features, hidden_units)
            self.fc2 = torch.nn.Linear(hidden_units, out_features)

        def forward(self, x):
            h = torch.nn.functional.tanh(self.fc1(x))
            return self.fc2(h)

        def set_parameters(self, z):
            offset = 0
            for m in self.children():

                weight_size = m.in_features * m.out_features
                m.weight.data = torch.Tensor(z[offset:offset + weight_size].reshape(m.out_features, m.in_features))
                offset += weight_size

                bias_size = m.out_features
                m.bias.data = torch.Tensor(z[offset:offset + bias_size])
                offset += bias_size

    for _ in range(trials):

        model_target = ANN()
        model = ANN()

        def f(z):

            model.set_parameters(z)

            loss = 0
            for x in 2 * torch.rand(inner_iterations, in_features) - 1:
                target = torch.autograd.Variable(model_target(x), requires_grad=False)
                loss += criterion(model(x), target)

            return -loss / inner_iterations

        param_count = in_features * hidden_units + hidden_units + hidden_units * out_features + out_features
        mu = np.random.randn(param_count)
        sigma = np.ones(param_count)

        res = snes.optimize(f, mu, sigma, record_history=True,
                            max_iter=100, rng=SEED, optimizer=torch.optim.SGD)

        assert(abs(np.mean(res['history_fitness'][-1])) < 0.1)
