"""Predicates for calculating split entropy."""

from typing import Callable

import numpy as np


def gini(x_data: np.ndarray) -> float:
    """Считает коэффициент Джини для массива меток x_data."""
    _, freqs = np.unique(x_data, return_counts=True)
    probs = freqs / x_data.shape[0]

    return 1 - np.sum(np.power(probs, 2))


def entropy(x_data: np.ndarray) -> float:
    """Считает энтропию для массива меток x_data."""
    _, freqs = np.unique(x_data, return_counts=True)
    probs = freqs / x_data.shape[0]

    return -np.sum(probs * np.log2(probs))


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
    y_data = np.hstack((left_y, right_y))
    return (
        len(y_data) * criterion(y_data)
        - len(left_y) * criterion(left_y)
        - len(right_y) * criterion(right_y)
    )
