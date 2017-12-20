"""Basic math tools
"""
import math
import functools

nan = math.nan


def list_check(default_return=nan):
    """
    annotation function for checking whether arguments whose type is list is None or empty
    :param default_return: lambda, if there is a parameter that is None or empty list, the function will return
    the result of `default_return`
    :return: decorate -> the decorated method
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in args:
                if not i and isinstance(i, list) and len(i) == 0:
                    return default_return
            return func(*args, **kwargs)

        return wrapper

    return decorate


def __mean__(*args, strict=False, return_all=False):
    """
    calculate the mean. This function is built for using the lib in REPL
    :return: the mean of data in `*args`
    :usage:
    >>> import statkit.basic as basic
    >>> basic.__mean__(1,2,3,4)
    2.5
    """
    if len(args) == 1:
        args = args[0]
    return mean(args, strict=strict, return_all=return_all)


@list_check()
def mean(lst, strict=False, return_all=False):
    """
    Calculate the mean in `lst`
    :param lst: the data set
    :param strict: if `strict` is true, the type of data in `lst` must be <number>; otherwise <str> can be accepted and
    will be automatically converted to <number>
    :param return_all: if `return_all` is true, the function will include the `lst` and the length(size) of the list
    otherwise, only the mean will be returned
    :return: the mean in `lst`
    :usage:
    >>> import statkit.basic as basic
    >>> basic.mean([1,2,3,4])
    2.5
    >>> basic.mean([1,2,3,'4'], strict=True)
    2.5
    >>> basic.mean([1,2,3,4], return_all=True)
    (2.5, [1.0, 2.0, 3.0, 4.0], 4)
    """
    try:
        lst = list(map(lambda x: float(x), lst))
    except ValueError:
        if strict:
            raise ValueError('Unexpected type')
        else:
            lst = list(
                filter(
                    lambda x: isinstance(x, int)
                              or isinstance(x, float)
                              or True if (isinstance(x, str) and x.isdigit()) else False, lst
                )
            )
            lst = list(map(lambda x: float(x), lst))
    return sum(lst) / len(lst) if not return_all else (sum(lst) / len(lst), lst, len(lst))


def __var__(*args, strict=False, is_population=False):
    return var(args, strict, is_population)


def var(lst, strict=False, is_population=False):
    """
    Calculate the variance of data in `lst`
    :param lst: the data set
    :param strict: see `help(basic.mean)`
    :param is_population: indicate whether the data is a set of population data
    :return: the variance of `lst`
    :usage:
    >>> import statkit.basic as basic
    >>> basic.var([1,2,3,4])
    1.6666666666666667
    >>> basic.var([1,2,3,4], is_population=True)
    1.25
    """
    mu, data, size = mean(lst, strict=strict, return_all=True)
    if mu == nan and data == nan and size == nan:
        return nan
    if size == 1:
        return 0
    if not is_population:
        size -= 1
    ret = sum(
        map(
            lambda x: (x - mu) ** 2, data
        )
    ) / size
    return ret


def __sd__(*args, strict=False, is_population=False):
    return sd(args, strict, is_population)


def sd(lst, strict=False, is_population=False):
    """
    Calculate the standard deviation of the data in `lst`
    :param lst: the data set
    :param strict: see `help(basic.mean)`
    :param is_population: see `help(basic.var)`
    :return: the standard deviation
    """
    variance = var(lst, strict=strict, is_population=is_population)
    return math.sqrt(variance)


def __mode__(*args, return_all=False):
    return mode(args, return_all)


@list_check()
def mode(lst, return_all=False):
    """
    Find the mode of `lst`
    :param lst: the data set
    :param return_all: if `return_all` is True, the function will return the mode and its number of occurrence
    :return: the mode
    :usage:
    >>> import statkit.basic as basic
    >>> basic.mode([1,1,2,3,4])
    1
    >>> basic.mode([1,1,2,2,3,4])
    (1,2)
    >>> basic.mode([1,1,2,2,3,4], return_all=True)
    ((1,2), 2)
    """
    _m = {}
    for i in lst:
        if i in _m.keys():
            _m[i] += 1
        else:
            _m[i] = 1
    if tuple(_m.values()).count(list(_m.values())[0]) == len(_m):
        return math.nan
    ret = list(_m.items())
    ret.sort(key=lambda x: -x[1])
    ans = list()
    _append = ans.append
    _t = None
    for i in enumerate(ret):
        if i[0] == 0:
            _append(i[1][0])
            _t = i[1][1]
        elif i[1][1] == _t:
            _append(i[1][0])
        else:
            break
    ans = tuple(ans)
    if return_all:
        return (ans[0], _m[ans[0]]) if len(ans) == 1 else (ans, _m[ans[0]])
    return ans[0] if len(ans) == 1 else ans


def z_score(x, lst):
    """
    Calculate the standard score for `x` in `lst`
    :param x: target variable
    :param lst: ultimate set of data
    :return: the standard score of `x`
    """
    if not lst or not len(lst):
        return nan
    mu = mean(lst)
    sdv = sd(lst)
    return (x - mu) / sdv


@list_check()
def b_range(lst):
    """
    calculate the range of `lst`
    :param lst: the ultimate data set
    :return:
    """
    return max(lst) - min(lst)


def __b_range__(*args):
    return b_range(args)


@list_check()
def median(lst):
    lst.sort()
    size = len(lst)
    if size & 1:
        return lst[size >> 1]
    else:
        return (lst[(size >> 1) - 1] + lst[size >> 1]) / 2


def __median__(*args):
    return median(args)


@list_check()
def iqr(lst, return_all=False):
    q_2 = median(lst)
    lst.sort()
    q_1 = median(
        list(filter(lambda x: x < q_2, lst))
    )
    q_3 = median(
        list(filter(lambda x: x > q_2, lst))
    )
    if return_all:
        return {
            'Q1': q_1,
            'Q2': q_2,
            'Q3': q_3,
            'IQR': q_3 - q_1,
        }
    return q_3 - q_1


def __iqr__(*args, return_all=False):
    return iqr(list(args), return_all)


def combination(n, m):
    if n < m:
        return nan
    if not m:
        return 1
    # return math.factorial(n) // (math.factorial(m) * math.factorial(n - m))
    return functools.reduce(lambda x, y: x * y, range(m + 1, n + 1)) // \
           functools.reduce(lambda x, y: x * y, range(1, n - m + 1))


def permutation(n, m):
    if n < m:
        return nan
    if not m:
        return 1
    # return math.factorial(n) // math.factorial(n - m)
    return functools.reduce(lambda x, y: x * y, range(n - m + 1, n + 1))


def incbeta(x, a, b):
    if x < 0.0 or x > 1.0:
        return nan
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - incbeta(1.0 - x, b, a)
    lbeta_ab = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta_ab)
    f = c = 1.0
    d = 0.0
    for i in range(0, 201):
        m = i >> 1
        numerator = 0.0
        if i == 0:
            numerator = 1.0
        elif i & 1:
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1))

        d = 1.0 + numerator * d
        if math.fabs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d

        c = 1.0 + numerator / c
        if math.fabs(c) < 1e-30:
            c = 1e-30

        cd = c * d
        f *= cd
        if math.fabs(1.0 - cd) < 1e-8:
            return front * (f - 1.0)
    return math.inf
