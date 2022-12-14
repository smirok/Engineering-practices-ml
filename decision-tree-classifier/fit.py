"""Entrypoint."""
import pickle

import numpy as np
from data.preprocessing import read_dataset
from dvclive import Live
from sklearn.model_selection import cross_val_score, train_test_split
from src.decision_tree_classifier import DecisionTreeClassifier


def fit():
    """Search the best decision tree hyperparameters."""
    x, y = read_dataset("data/dataset/train.csv")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=42
    )

    best_cvs = 0
    best_params = None
    for criterion in ["gini", "entropy"]:
        for max_depth in [3, 5, 7]:
            for min_samples_leaf in [3, 5, 10, 25]:
                dtc = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                )
                cvs = cross_val_score(dtc, x_train, y_train, cv=5)

                if np.mean(cvs) > best_cvs:
                    best_cvs = np.mean(cvs)
                    best_params = {
                        "criterion": criterion,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    }

    with Live("evaluation") as live:
        live.log_params(best_params)

    bestclf = DecisionTreeClassifier(
        criterion=best_params["criterion"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
    )

    bestclf.fit(x_train, y_train)
    with open("data/model/model.pkl", "wb+") as fd:
        pickle.dump(bestclf, fd)


if __name__ == "__main__":
    fit()
