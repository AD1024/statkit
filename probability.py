"""Probability calculation and distribution calculation
"""
import statkit.basic as _basic
import math as _math


@_basic.list_check()
def expected(_x, _p):
    if len(_x) != len(_p):
        return _basic.nan
    return sum(map(lambda i: _x[i] * _p[i], range(len(_x))))


@_basic.list_check()
def var(_x, _p):
    if len(_x) != len(_p):
        return _basic.nan
    mu = expected(_x, _p)
    return sum(map(lambda i: _p[i] * (_x[i] - mu) ** 2), range(len(_x)))


class dist:
    @classmethod
    def norm_pd(cls, _x, sd, mu):
        return (1 / (sd * _math.sqrt(2 * _math.pi))) * (_math.e ** (-((_x - mu) ** 2) / (2 * (sd ** 2))))

    @classmethod
    def _cdf(cls, x, sd, mu):
        return 0.5 * (1 + _math.erf((x - mu) / (sd * _math.sqrt(2))))

    @classmethod
    def norm_cd(cls, sd, mu, lower=-_math.inf, upper=_math.inf):
        lower, upper = (lower, upper) if lower <= upper else (upper, lower)
        return cls._cdf(upper, sd, mu) - cls._cdf(lower, sd, mu)

    # @classmethod
    # def inv_norm_cd(cls, x, sd, mu):
    #     return _math.sqrt((sd / (2 * _math.pi * (x ** 3)))) * _math.exp((-sd * (x - mu) ** 2) / 2 * (mu ** 2) * x)

    @classmethod
    def plot_norm_curve(cls, sd, mu, step=100):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError('mathplotlib is not installed')
        _x = np.linspace(mu - 4 * sd, mu + 4 * sd + 1, step)
        func = lambda x: (1 / (sd * _math.sqrt(2 * _math.pi))) * (_math.e ** (-((x - mu) ** 2) / (2 * (sd ** 2))))
        _y = list(map(func, _x))
        plt.plot(_x, _y)
        plt.show()
        plt.close()

    @classmethod
    def binom_pd(cls, cnt, size, prob):
        if size == 0 or cnt > size:
            return _basic.nan
        return _basic.combination(size, cnt) * (prob ** cnt) * ((1.0 - prob) ** (size - cnt))

    @classmethod
    def binom_cd(cls, cnt, size, prob):
        if size == 0 or cnt > size:
            return _basic.nan
        ans = 0
        for i in range(0, cnt+1):
            ans += cls.binom_pd(i, size, prob)
        return ans

