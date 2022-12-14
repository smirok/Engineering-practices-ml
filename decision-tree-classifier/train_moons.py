"""Entrypoint."""
import pickle

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier


def train_moons():
    """Moons classification."""
    noise = 0.35
    x_train, y_train = make_moons(1500, noise=noise, random_state=42)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30)
    tree.fit(x_train, y_train)

    with open("data/model/tree_moons.pkl", "wb+") as fd:
        pickle.dump(tree, fd)


if __name__ == "__main__":
    train_moons()
