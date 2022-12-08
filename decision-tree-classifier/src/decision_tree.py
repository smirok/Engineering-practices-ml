"""Decision tree components."""

from typing import Union

import numpy as np


class DecisionTreeLeaf:
    """
    Class for storing data in tree leaf.

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса,
        который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, ys, classes):
        """
        Decision tree leaf initialization.

        Parameters
        ----------
        ys : np.ndarray
            Метки элементов в листе
        classes : np.ndarray
            Классы элементов в дереве
        """
        keys, counts = np.unique(ys, return_counts=True)
        self.probs = dict(zip(classes, np.zeros(len(classes))))
        for key, cnt in zip(keys, counts):
            self.probs[key] = cnt / len(ys)
        self.y = keys[np.argmax(counts)]


class DecisionTreeNode:
    """
    Class for storing data in tree node.

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """

    def __init__(
        self,
        split_dim: int,
        split_value: float,
        left: Union["DecisionTreeNode", DecisionTreeLeaf],
        right: Union["DecisionTreeNode", DecisionTreeLeaf],
    ):
        """
        Decision tree node initialization.

        Parameters
        ----------
        split_dim : int
            Измерение, по которому разбиваем выборку.
        split_value : float
            Значение, по которому разбираем выборку.
        left : Union[DecisionTreeNode, DecisionTreeLeaf]
            Поддерево, отвечающее за случай x[split_dim] < split_value.
        right : Union[DecisionTreeNode, DecisionTreeLeaf]
            Поддерево, отвечающее за случай x[split_dim] >= split_value.
        """
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
