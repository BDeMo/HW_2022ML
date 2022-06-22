import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import NLLLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
import aim
from tqdm import trange

import config

from model.Net import Mclr_Logistic, DNN, lnr, lls
from utils.dec_utils import to_one_hot

SEED = 2022
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

ModelList = {
    'mclr': Mclr_Logistic
    , 'dnn': DNN
    , 'lnr': lnr
    , 'lls': lls
}
LossList = {
    'NLLLoss': NLLLoss
    , 'CrossEntropyLoss': CrossEntropyLoss
    , 'MSELoss': MSELoss
}

if __name__ == "__main__":
    print('Loading data')
    if config.dec:
        print(config.dec_data_path)
        datas = np.load(config.dec_data_path)
    else:
        print(config.data_path)
        datas = np.load(config.data_path)
    print(config.label_path)
    labels = np.load(config.label_path)
    datas = [(Tensor(x).type(torch.float32), y) for x, y in zip(datas[:], labels[:])]
    iterdata = iter(DataLoader(datas, batch_size=config.num_train, shuffle=True))
    d_train, l_train = next(iterdata)
    d_test, l_test = next(iterdata)

    print('Data Loaded')
    train_data = [(x, y) for x, y in zip(d_train, l_train)]
    test_data = [(x, y) for x, y in zip(d_test, l_test)]
    trainloader = DataLoader(train_data, config.batchsize, shuffle=True)
    testloader = DataLoader(test_data, config.batchsize, shuffle=True)
    itertrain = iter(trainloader)
    # itertest = iter(testloader)

    # trainloader_full = DataLoader(train_data, len(train_data), shuffle=True)
    testloader_full = DataLoader(test_data, len(test_data), shuffle=True)

    device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() and config.gpu != -1 else "cpu")

    if config.LLSM:

        run = aim.Run(experiment=config.exp_name + '_linear_MSE_dec')
        run['hparam'] = {
            'num_train': config.num_train,
            'num_test': config.num_test,
            'learning_rate': config.learning_rate,
            'batchsize': config.batchsize,
            'num_epochs': config.num_epochs,
            'model': 'linear',
            'loss': 'MSELoss'
        }

        X = d_train
        Y = to_one_hot(l_train, n_class=2)

        ones_2_insert = np.ones((1, X.shape[0],))
        X = np.insert(X, 0, values=ones_2_insert, axis=1)
        res = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
        b = res[0, :]
        A = res[1:, :].T
        print(b.shape, A.shape)

        X_test = d_test
        ones_2_insert = np.ones((1, X_test.shape[0],))
        print(X_test.shape, ones_2_insert.shape)
        X_test = np.insert(X_test, 0, values=ones_2_insert, axis=1)
        Y_test = to_one_hot(l_test, n_class=2)
        # print(Y_test.shape)
        # print(np.linalg.norm(Y_test))
        # print(np.dot(X_test, res))
        #
        # print(np.linalg.norm(Y_test - np.dot(X_test, res)))

        train_acc = 0.
        train_loss = 0.
        for x, y in trainloader:
            X = np.insert(x, 0, values=np.ones((1, x.shape[0],)), axis=1)
            Y = Tensor(to_one_hot(y))
            output = Tensor(np.dot(X, res))
            train_acc += (torch.sum(torch.argmax(output, dim=1) == torch.argmax(Y, dim=1)))
            train_loss += np.linalg.norm(output - Y)
        print(train_acc / config.num_train, train_loss / config.num_train)

        test_acc = 0.
        test_loss = 0.
        for x, y in testloader_full:
            X = np.insert(x, 0, values=np.ones((1, x.shape[0],)), axis=1)
            Y = Tensor(to_one_hot(y))
            output = Tensor(np.dot(X, res))
            test_acc += (torch.sum(torch.argmax(output, dim=1) == torch.argmax(Y, dim=1)))
            test_loss += np.linalg.norm(output - Y)
        print(test_acc / config.num_test, test_loss / config.num_test)

        for i in trange(config.num_epochs):
            run.track(train_acc / config.num_train, 'Training Accuracy(%)', epoch=i)
            run.track(train_loss / config.num_train, 'Training Loss', epoch=i)
            run.track(test_acc / config.num_test, 'Testing Accuracy(%)', epoch=i)
            run.track(test_loss / config.num_test, 'Testing Loss', epoch=i)

    else:
        print('Training')
        for ml in config.mandl:
            m, l = ml
            for t in range(config.times):
                if config.dec:
                    tag = '_dec'
                    model = ModelList[m](input_dim=config.dec_dim).to(device)
                else:
                    tag = ''
                    model = ModelList[m]().to(device)
                loss = LossList[l]()
                optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

                run = aim.Run(experiment=config.exp_name + '_' + m + '_' + l + tag)
                run['hparam'] = {
                    'num_train': config.num_train,
                    'num_test': config.num_test,
                    'learning_rate': config.learning_rate,
                    'batchsize': config.batchsize,
                    'num_epochs': config.num_epochs,
                    'model': m,
                    'loss': l
                }
                print(m, l, 'times', t)
                for i in trange(config.num_epochs):
                    while (True):
                        try:
                            (x, y) = next(itertrain)
                        except StopIteration:
                            itertrain = iter(trainloader)
                            break
                        if l == "MSELoss":
                            X, Y = x.to(device), torch.eye(config.NUM_CLASSES)[y, :].to(device)
                        else:
                            X, Y = x.to(device), y.to(device)
                        model.train()
                        optimizer.zero_grad()
                        output = model(X)
                        _loss = loss(output, Y)
                        _loss.backward()
                        optimizer.step()

                    model.eval()

                    train_acc = 0.
                    train_loss = 0.
                    for x, y in trainloader:
                        if l == "MSELoss":
                            X, Y = x.to(device), torch.eye(config.NUM_CLASSES)[y, :].to(device)
                        else:
                            X, Y = x.to(device), y.to(device)
                        output = model(X)
                        if l == "MSELoss":
                            train_acc += (torch.sum(torch.argmax(output, dim=1) == torch.argmax(Y, dim=1))).item()
                        else:
                            train_acc += (torch.sum(torch.argmax(output, dim=1) == Y)).item()
                        train_loss += loss(output, Y).item()
                    run.track(train_acc / config.num_train, 'Training Accuracy(%)', epoch=i)
                    run.track(train_loss / config.num_train, 'Training Loss', epoch=i)

                    test_acc = 0.
                    test_loss = 0.
                    for x, y in testloader_full:
                        if l == "MSELoss":
                            X, Y = x.to(device), torch.eye(config.NUM_CLASSES)[y, :].to(device)
                        else:
                            X, Y = x.to(device), y.to(device)
                        output = model(X)
                        if l == "MSELoss":
                            test_acc += (torch.sum(torch.argmax(output, dim=1) == torch.argmax(Y, dim=1))).item()
                        else:
                            test_acc += (torch.sum(torch.argmax(output, dim=1) == Y)).item()
                        test_loss += loss(output, Y).item()
                    run.track(test_acc / config.num_test, 'Testing Accuracy(%)', epoch=i)
                    run.track(test_loss / config.num_test, 'Testing Loss', epoch=i)
