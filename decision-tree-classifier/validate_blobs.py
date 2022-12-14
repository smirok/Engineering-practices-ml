"""Entrypoint."""
import pickle

from sklearn.datasets import make_blobs


def validate_blobs():
    """Blobs classification."""
    x_test, y_test = make_blobs(10, 1, centers=[[5, 0], [-10, 0]], random_state=42)

    with open("data/model/tree_blobs.pkl", "rb") as fd:
        tree = pickle.load(fd)

    print(tree.score(x_test, y_test))


if __name__ == "__main__":
    validate_blobs()
