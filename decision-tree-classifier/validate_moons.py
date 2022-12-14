"""Entrypoint."""
import pickle

from sklearn.datasets import make_moons


def validate_moons():
    """Moons classification."""
    noise = 0.35
    x_test, y_test = make_moons(200, noise=noise, random_state=42)

    with open("data/model/tree_moons.pkl", "rb") as fd:
        tree = pickle.load(fd)

    print(tree.score(x_test, y_test))


if __name__ == "__main__":
    validate_moons()
