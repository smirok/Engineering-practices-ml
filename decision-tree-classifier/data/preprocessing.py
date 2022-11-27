import numpy as np
import pandas
from sklearn.utils import shuffle


def read_dataset(path):
    dataframe = pandas.read_csv(path, header=1)
    dataset = dataframe.values.tolist()
    shuffle(dataset, random_state=42)

    y = [row[0] for row in dataset]
    X = [row[1:] for row in dataset]
    return np.array(X), np.array(y)
