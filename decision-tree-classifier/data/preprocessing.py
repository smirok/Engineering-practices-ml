"""Reading the dataset file."""

import numpy as np
import pandas
from sklearn.utils import shuffle


def read_dataset(path):
    """Read the dataset file by path."""
    dataframe = pandas.read_csv(path, header=1)
    dataset = dataframe.values.tolist()
    shuffle(dataset, random_state=42)

    y_data = [row[0] for row in dataset]
    x_data = [row[1:] for row in dataset]
    return np.array(x_data), np.array(y_data)
