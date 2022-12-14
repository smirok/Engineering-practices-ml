"""Entrypoint."""
import pickle

from sklearn.datasets import make_blobs


def validate_blobs_hard():
    """Multiple blob clusters classification."""
    x_test, y_test = make_blobs(
        200, 2, centers=[[0, 0], [-2.5, 0], [3, 2], [1.5, -2.0]], random_state=42
    )

    with open("data/model/tree_blobs_hard.pkl", "rb") as fd:
        tree = pickle.load(fd)

    print(tree.score(x_test, y_test))


if __name__ == "__main__":
    validate_blobs_hard()
