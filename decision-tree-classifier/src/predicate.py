from typing import Callable

import numpy as np


def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """

    n = x.shape[0]
    _, freqs = np.unique(x, return_counts=True)
    probs = freqs / n

    return 1 - np.sum(np.power(probs, 2))


def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """

    n = x.shape[0]
    _, freqs = np.unique(x, return_counts=True)
    probs = freqs / n

    return - np.sum(probs * np.log2(probs))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """

    y = np.hstack((left_y, right_y))
    return len(y) * criterion(y) - len(left_y) * criterion(left_y) - len(right_y) * criterion(right_y)
