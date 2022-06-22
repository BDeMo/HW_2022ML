import random

import numpy as np
import torch

import config
from utils.dec_utils import to_one_hot

SEED = 2022
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    print('Loading data')
    print(config.dec_data_path)
    print(config.label_path)
    datas = np.load(config.dec_data_path)
    labels = np.load(config.label_path)

    print('Data Loaded')

    X = datas
    Y = to_one_hot(labels, n_class=2)

    ones_2_insert = np.ones((1, X.shape[0],))
    X = np.insert(X, 0, values=ones_2_insert, axis=1)
    res = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    b = res[0, :]
    A = res[1:, :].T
    print(b.shape, A.shape)
