import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

from data.preprocessing import read_dataset
from src.decision_tree_classifier import DecisionTreeClassifier


def solve_blobs():
    X, y = make_blobs(50, 1, centers=[[5, 0], [-10, 0]])
    X_test, y_test = make_blobs(10, 1, centers=[[5, 0], [-10, 0]])
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5)
    tree.fit(X, y)
    print(tree.score(X_test, y_test))


def solve_moons():
    noise = 0.35
    X, y = make_moons(1500, noise=noise)
    X_test, y_test = make_moons(200, noise=noise)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30)
    tree.fit(X, y)
    print(tree.score(X_test, y_test))


def solve_blobs_hard():
    X, y = make_blobs(1500, 2, centers=[[0, 0], [-2.5, 0], [3, 2], [1.5, -2.0]])
    X_test, y_test = make_blobs(
        200, 2, centers=[[0, 0], [-2.5, 0], [3, 2], [1.5, -2.0]]
    )
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
    tree.fit(X, y)
    print(tree.score(X_test, y_test))


def best_fit():
    X, y = read_dataset("data/dataset/train.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    best_cvs = 0
    best_params = None
    for max_depth in [3, 5, 7]:
        for min_samples_leaf in [3, 5, 10, 25]:
            dtc = DecisionTreeClassifier(
                criterion="gini", max_depth=max_depth, min_samples_leaf=min_samples_leaf
            )
            cvs = cross_val_score(dtc, X_train, y_train, cv=5)

            if np.mean(cvs) > best_cvs:
                best_cvs = np.mean(cvs)
                best_params = {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                }

    bestclf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
    )

    bestclf.fit(X_train, y_train)
    y_pred = bestclf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(best_params)
    print(best_cvs)


if __name__ == "__main__":
    solve_blobs()
    solve_moons()
    solve_blobs_hard()
    best_fit()
