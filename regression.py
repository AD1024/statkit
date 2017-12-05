from .basic import *


@list_check()
def lr(_x, _y):
    if len(_x) != len(_y):
        return nan
    mu_x = mean(_x)
    mu_y = mean(_y)
    f = {
        'x': lambda i: _x[i] - mu_x,
        'y': lambda i: _y[i] - mu_y,
    }
    size = len(_x)
    return sum(
        map(
            lambda i: f['x'](i) * f['y'](i),
            range(size)
        )
    ) / math.sqrt(
        sum(map(lambda i: f['x'](i) ** 2, range(size)))
    ) / math.sqrt(
        sum(map(lambda i: f['y'](i) ** 2, range(size)))
    )


def lr_sqr(_x, _y):
    return lr(_x, _y) ** 2


@list_check()
def lm(_x, _y, return_all=False):
    if len(_x) != len(_y):
        return nan
    f = {
        'x': lambda i: _x[i] - mu_x,
        'y': lambda i: _y[i] - mu_y,
    }
    size = len(_x)
    mu_x = mean(_x)
    mu_y = mean(_y)
    b_1 = sum(
        map(lambda i: f['x'](i) * f['y'](i), range(size))
    ) / sum(
        map(lambda i: f['x'](i) ** 2, range(size))
    )
    b_0 = mu_y - b_1 * mu_x
    if return_all:
        return {
            'formula': 'y={a}x+{b}'.format(a=b_1, b=b_0),
            'a': b_1,
            'b': b_0,
            'r-sqr': lr_sqr(_x, _y),
            'func': lambda x: x * b_1 + b_0,
        }
    else:
        return lambda x: x * b_1 + b_0


@list_check()
def plot_reg_line(_x, _y, reg=lm):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise e
    func = reg(_x, _y)
    _yhat = list(map(lambda x: func(x), _x))
    plt.scatter(_x, _y)
    plt.plot(_x, _yhat)
    plt.show()
    plt.close()


@list_check()
def residual(_x, _y, reg=lm):
    func = reg(_x, _y)
    return list(map(lambda i: _y[i] - func(_x[i]), range(len(_x))))


@list_check()
def plot_residual(_x, _y, res=residual):
    resd = res(_x, _y)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('matplotlib is not installed')
    plt.scatter(_x, resd)
    plt.show()
    plt.close()
