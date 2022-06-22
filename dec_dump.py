import random

import numpy as np
import torch
from sklearn import decomposition

import config

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

    pca = decomposition.PCA(n_components=1000)
    X = pca.fit_transform(X)
    print(pca.n_components_)
    print(X.shape)
    np.save(config.dec_data_path, X)
