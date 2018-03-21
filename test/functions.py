def f_1d(x, x0):
    return -1. * ((x - x0) ** 2)


def f_2d(x, x0, y0):
    return -1. * ((x[0] - x0) ** 2 + (x[1] - y0) ** 2)


def f_2d_nonisotropic(x, x0, y0):
    return -1. * ((x[0] - x0) ** 2 + 0.01 * (x[1] - y0) ** 2)


def f_rosenbrock(x, a, b):
    return -1. * ((a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2)
