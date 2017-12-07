import statkit.basic as basic
import math


@basic.list_check()
def expected(_x, _p):
    if len(_x) != len(_p):
        return basic.nan
    return sum(map(lambda i: _x[i] * _p[i], range(len(_x))))


@basic.list_check()
def var(_x, _p):
    if len(_x) != len(_p):
        return basic.nan
    mu = expected(_x, _p)
    return sum(map(lambda i: _p[i] * (_x[i] - mu) ** 2), range(len(_x)))


class Dist:
    @classmethod
    def norm_pd(cls, _x, sd, mu):
        return (1 / (sd * math.sqrt(2 * math.pi))) * (math.e ** (-((_x - mu) ** 2) / (2 * (sd ** 2))))

    @classmethod
    def _cdf(cls, x, sd, mu):
        return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))

    @classmethod
    def norm_cd(cls, sd, mu, lower=-math.inf, upper=math.inf):
        lower, upper = (lower, upper) if lower <= upper else (upper, lower)
        return cls._cdf(upper, sd, mu) - cls._cdf(lower, sd, mu)

    @classmethod
    def plot_norm_curve(cls, sd, mu, step=100):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError('mathplotlib is not installed')
        _x = np.linspace(mu - 4 * sd, mu + 4 * sd + 1, step)
        func = lambda x: (1 / (sd * math.sqrt(2 * math.pi))) * (math.e ** (-((x - mu) ** 2) / (2 * (sd ** 2))))
        _y = list(map(func, _x))
        plt.plot(_x, _y)
        plt.show()
        plt.close()

    @classmethod
    def binom_pd(cls, cnt, size, prob):
        if size == 0 or cnt > size:
            return basic.nan
        return basic.combination(size, cnt) * (prob ** cnt) * ((1.0 - prob) ** (size - cnt))

    @classmethod
    def binom_cd(cls, cnt, size, prob):
        if size == 0 or cnt > size:
            return basic.nan
        ans = 0
        for i in range(0, cnt+1):
            ans += cls.binom_pd(i, size, prob)
        return ans

