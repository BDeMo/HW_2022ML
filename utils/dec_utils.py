import numpy as np


def to_one_hot(labels, n_class=2):
    res = np.eye(n_class)[labels]
    return res
