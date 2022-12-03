"""Decision tree classifier."""

from copy import deepcopy
from typing import Any, Dict, List, NoReturn, Optional, Union

import numpy as np
from sklearn.metrics import accuracy_score
from src.decision_tree import DecisionTreeLeaf, DecisionTreeNode
from src.predicate import entropy, gain, gini


class DecisionTreeClassifier:
    """
    Class for classification using decision tree.

    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
    ):
        """
        Classifier initialization.

        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева
            Возможные значения: "gini", "entropy"
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева
        """
        self.root = None
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        x_train : np.ndarray
            Обучающая выборка.
        y_train : np.ndarray
            Вектор меток классов.
        """
        self.classes = np.unique(y_train)
        self.root = self._build_(x_train, y_train, np.arange(len(y_train)))

    def _build_(
        self, x_data: np.ndarray, y_data: np.ndarray, indices: np.ndarray, depth=0
    ) -> Union[DecisionTreeNode, DecisionTreeLeaf]:
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeLeaf(y_data[indices], self.classes)

        n_samples, n_features = len(indices), x_data.shape[1]

        best_ig = 0
        split_dim, split_value, split_pos = None, None, None
        for dim in range(n_features):
            sorted_indices = (
                np.array(
                    sorted(
                        np.vstack((x_data[indices, dim], indices)).T, key=lambda x: x[0]
                    )
                )[:, 1]
            ).astype(int)

            for i in range(n_samples - 1):  # split by <=
                ig = gain(
                    y_data[sorted_indices[: (i + 1)]],
                    y_data[sorted_indices[(i + 1) :]],
                    self.criterion,
                )

                if (
                    ig > best_ig
                    and min(i + 1, n_samples - i - 1) >= self.min_samples_leaf
                ):
                    split_dim = dim
                    split_pos = i + 1
                    split_value = x_data[sorted_indices[split_pos], split_dim]
                    best_ig = ig

        if best_ig > 0:
            sorted_indices = (
                np.array(
                    sorted(
                        np.vstack((x_data[indices, split_dim], indices)).T,
                        key=lambda x: x[0],
                    )
                )[:, 1]
            ).astype(int)

            return DecisionTreeNode(
                split_dim=split_dim,
                split_value=split_value,
                left=self._build_(
                    x_data, y_data, sorted_indices[:split_pos], depth + 1
                ),
                right=self._build_(
                    x_data, y_data, sorted_indices[split_pos:], depth + 1
                ),
            )

        return DecisionTreeLeaf(y_data[indices], self.classes)

    def predict_proba(self, x_data: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из x_data.

        Parameters
        ----------
        x_data : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из x_data возвращает словарь
            {метка класса -> вероятность класса}.
        """
        ans = []
        for x_elem in x_data:
            node = deepcopy(self.root)

            while isinstance(node, DecisionTreeNode):
                if x_elem[node.split_dim] <= node.split_value:
                    node = node.left
                else:
                    node = node.right

            ans.append(node.probs)
        return ans

    def predict(self, x_data: np.ndarray) -> list:
        """
        Предсказывает классы для элементов x_data.

        Parameters
        ----------
        x_data : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов x_data.
        """
        proba = self.predict_proba(x_data)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]

    def get_params(self, deep=False):
        """Возвращает параметры модели."""
        deep
        return {
            "criterion": "gini" if self.criterion == gini else "entropy",
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
        }

    def score(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        Считает accuracy score для элементов x_data и исходных меток y_data.

        Parameters
        ----------
        x_data : np.ndarray
            Элементы для предсказания.
        y_data: np.ndarray
            Исходные метки элементов.

        Return
        ------
        list
            Вектор предсказанных меток для элементов x_data.
        """
        return accuracy_score(y_data, self.predict(x_data))
