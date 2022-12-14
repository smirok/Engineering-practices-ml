"""Entrypoint."""
import pickle

from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier


def train_blobs_hard():
    """Multiple blob clusters classification."""
    x_train, y_train = make_blobs(
        1500, 2, centers=[[0, 0], [-2.5, 0], [3, 2], [1.5, -2.0]], random_state=42
    )
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
    tree.fit(x_train, y_train)

    with open("data/model/tree_blobs_hard.pkl", "wb+") as fd:
        pickle.dump(tree, fd)


if __name__ == "__main__":
    train_blobs_hard()
