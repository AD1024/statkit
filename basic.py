"""Basic math tools
"""
import math
import functools

nan = math.nan


def list_check(default_return=lambda: nan):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in args:
                if not i and isinstance(i, list) and len(i) == 0:
                    return default_return()
            return func(*args, **kwargs)

        return wrapper

    return decorate


def __mean__(*args, strict=False, return_all=False):
    if len(args) == 1:
        args = args[0]
    return mean(args, strict=strict, return_all=return_all)


@list_check()
def mean(lst, strict=False, return_all=False):
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
    variance = var(lst, strict=strict, is_population=is_population)
    return math.sqrt(variance)


def __mode__(*args, return_all=False):
    return mode(args, return_all)


@list_check()
def mode(lst, return_all=False):
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
    if return_all:
        return (ans[0], _m[ans[0]]) if len(ans) == 1 else (ans, _m[ans[0]])
    return ans[0] if len(ans) == 1 else ans


def z_score(x, lst):
    if not lst or not len(lst):
        return nan
    mu = mean(lst)
    sdv = sd(lst)
    return (x - mu) / sdv


@list_check()
def b_range(lst):
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
def iqr(lst):
    q_2 = median(lst)
    lst.sort()
    q_1 = median(
        list(filter(lambda x: x < q_2, lst))
    )
    q_3 = median(
        list(filter(lambda x: x > q_2, lst))
    )
    return q_3 - q_1


def __iqr__(*args):
    return iqr(list(args))
