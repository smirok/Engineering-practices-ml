"""Entrypoint."""
import pickle

from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier


def train_blobs():
    """Blobs classification."""
    x_train, y_train = make_blobs(50, 1, centers=[[5, 0], [-10, 0]], random_state=42)
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5)
    tree.fit(x_train, y_train)

    with open("data/model/tree_blobs.pkl", "wb+") as fd:
        pickle.dump(tree, fd)


if __name__ == "__main__":
    train_blobs()
