"""Collection of the core mathematical operators used throughout the code base."""

import math
import typing as tp

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(a: float, b: float) -> float:
    return a * b


def id(a: float) -> float:
    return a


def add(a: float, b: float) -> float:
    return a + b


def neg(a: float) -> float:
    return -a


def lt(a: float, b: float) -> bool:
    return a < b


def eq(a: float, b: float) -> bool:
    return a == b


def is_close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9


def sigmoid(a: float) -> float:
    return 1 / (1 + math.exp(-a))


def relu(a: float) -> float:
    return a if a > 0 else 0


def log(a: float) -> float:
    return math.log(a)


def exp(a: float) -> float:
    return math.exp(a)


def inv(a: float) -> float:
    return 1 / a


def log_back(a: float, b: float) -> float:
    return b / a


def inv_back(a: float, b: float) -> float:
    return -b / (a * a)


def relu_back(a: float, b: float) -> float:
    return b if a > 0 else 0


def max(x: float, y: float) -> float:
    return x if x > y else y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: tp.Callable[[float], float]) -> tp.Callable[[tp.Iterable[float]], tp.Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    return lambda values: (fn(row) for row in values)


def zipWith(
    fn: tp.Callable
) -> tp.Callable:
    return lambda left_value, right_values: (
        fn(left_value, right_values)
        for left_value, right_values in zip(left_value, right_values)
    )


def reduce(func: tp.Callable) -> tp.Callable:
    def foo(values: tp.Iterable[float]) -> float:
        prev_val = None
        for val in values:
            if prev_val is None:
                prev_val = val
            else:
                prev_val = func(prev_val, val)
        if prev_val is None:
            return 0
        return prev_val
    return foo


def negList(values: tp.Iterable[float]) -> tp.Iterable[float]:
    return map(neg)(values)


def addLists(
    left_values: tp.Iterable[float], right_values: tp.Iterable[float]
) -> tp.Iterable[float]:
    return zipWith(add)(left_values, right_values)


def sum(values: tp.Iterable[float]) -> float:
    return reduce(add)(values)


def prod(values: tp.Iterable[float]) -> float:
    return reduce(mul)(values)
