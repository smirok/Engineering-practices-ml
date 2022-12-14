"""Entrypoint."""
import pickle

from data.preprocessing import read_dataset
from dvclive import Live
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def predict():
    """Predict using the best decision tree."""
    x, y = read_dataset("data/dataset/train.csv")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=42
    )

    with open("data/model/model.pkl", "rb") as fd:
        clf = pickle.load(fd)

    y_pred = clf.predict(x_test)

    with Live("evaluation") as live:
        accuracy = accuracy_score(y_test, y_pred)
        live.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    predict()
