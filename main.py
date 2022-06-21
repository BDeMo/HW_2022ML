import random

import numpy as np
import torch
from torch import Tensor
from torch.nn import NLLLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
import aim
from tqdm import trange

import config

from model.Net import Mclr_Logistic, DNN, lnr

SEED = 2022
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

NUM_CLASSES = 2

exp_name = 'hw_2022ml'
data_path = './data/data.npy'
label_path = './data/label.npy'

ModelList = {
    'mclr': Mclr_Logistic
    , 'dnn': DNN
    , 'linear': lnr
}
LossList = {
    'NLLLoss': NLLLoss
    , 'CrossEntropyLoss': CrossEntropyLoss
    , 'MSELoss': MSELoss
}

if __name__ == "__main__":
    print('Loading data')
    print(data_path)
    print(label_path)
    datas = np.load(data_path)
    labels = np.load(label_path)
    datas = [(Tensor(x).type(torch.float32), y) for x, y in zip(datas[:], labels[:])]
    iterdata = iter(DataLoader(datas, batch_size=config.num_train, shuffle=True))
    d_train, l_trian = next(iterdata)
    d_test, l_test = next(iterdata)

    print('Data Loaded')
    train_data = [(x, y) for x, y in zip(d_train, l_trian)]
    test_data = [(x, y) for x, y in zip(d_test, l_test)]
    trainloader = DataLoader(train_data, config.batchsize, shuffle=True)
    testloader = DataLoader(test_data, config.batchsize, shuffle=True)
    itertrain = iter(trainloader)
    # itertest = iter(testloader)

    # trainloader_full = DataLoader(train_data, len(train_data), shuffle=True)
    testloader_full = DataLoader(test_data, len(test_data), shuffle=True)

    device = torch.device("cuda:{}".format(config.gpu) if torch.cuda.is_available() and config.gpu != -1 else "cpu")

    print('Training')
    for ml in config.mandl:
        m, l = ml
        for t in range(config.times):
            model = ModelList[m]().to(device)
            loss = LossList[l]()
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

            run = aim.Run(experiment=exp_name + '_' + m + '_' + l)
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
                        X, Y = x.to(device), torch.eye(NUM_CLASSES)[y, :].to(device)
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
                        X, Y = x.to(device), torch.eye(NUM_CLASSES)[y, :].to(device)
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
                        X, Y = x.to(device), torch.eye(NUM_CLASSES)[y, :].to(device)
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
