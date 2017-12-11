"""Regression and regression analysis
"""
import statkit.basic as basic
import functools
import math as _math

_mean = basic.mean
_nan = basic.nan
_list_check = basic.list_check


def args_exclusive(ex_set=None):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kargs):
            for i in args:
                if i in ex_set:
                    return _nan
            return func(*args, **kargs)

        return wrapper

    return decorate


@_list_check()
def lr(_x, _y):
    if len(_x) != len(_y):
        return _nan
    mu_x = _mean(_x)
    mu_y = _mean(_y)
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
    ) / _math.sqrt(
        sum(map(lambda i: f['x'](i) ** 2, range(size)))
    ) / _math.sqrt(
        sum(map(lambda i: f['y'](i) ** 2, range(size)))
    )


def lr_sqr(_x, _y):
    return lr(_x, _y) ** 2


@_list_check()
def lm(_x, _y, return_all=False):
    if len(_x) != len(_y):
        return _nan
    f = {
        'x': lambda i: _x[i] - mu_x,
        'y': lambda i: _y[i] - mu_y,
    }
    size = len(_x)
    mu_x = _mean(_x)
    mu_y = _mean(_y)
    b_1 = sum(
        map(lambda i: f['x'](i) * f['y'](i), range(size))
    ) / sum(
        map(lambda i: f['x'](i) ** 2, range(size))
    )
    b_0 = mu_y - b_1 * mu_x
    if return_all:
        return {
            'formula': 'y={a}x+{b}'.format(a='%.3f' % b_1, b='%.3f' % b_0),
            'a': b_1,
            'b': b_0,
            'r-sqr': lr_sqr(_x, _y),
            'func': lambda x: x * b_1 + b_0,
        }
    else:
        return lambda x: x * b_1 + b_0


@_list_check()
def plot_reg_line(_x, _y, reg_func=None, reg=lm):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise e
    reg_info = reg(_x, _y, True) if not reg_func else reg_func
    func = reg_info['func']
    _yhat = list(map(lambda x: func(x), _x))
    plt.scatter(_x, _y, label=reg_info['formula'])
    plt.plot(_x, _yhat)
    plt.legend()
    plt.show()
    plt.close()


@_list_check()
def residual(_x, _y, reg=lm):
    func = reg(_x, _y)
    return list(map(lambda i: _y[i] - func(_x[i]), range(len(_x))))


@_list_check()
def plot_residual(_x, _y, res=residual):
    resd = res(_x, _y)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('matplotlib is not installed')
    plt.scatter(_x, resd)
    plt.show()
    plt.close()


@_list_check()
def sse(_x, _y, func):
    if len(_x) != len(_y):
        return _nan
    return sum(map(lambda i: (_y[i] - func(_x[i])), range(_x)))


@_list_check()
def mse(_x, _y, func):
    return sse(_x, _y, func) / len(_x)


@_list_check()
def ssr(_x, _y, func):
    mu_y = _mean(_y)
    return sum(map(lambda x: (func(x) - mu_y) ** 2, _x))


@_list_check()
def sst(_y):
    mu_y = _mean(_y)
    return sum(map(lambda x: (x - mu_y) ** 2, _y))


@args_exclusive((0,))
def se(_sse, size):
    return _math.sqrt(_sse / (size - 2))
