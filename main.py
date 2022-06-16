import numpy as np
import torch
from torch import Tensor
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
import aim
from tqdm import trange

import config
from config import *

from model.Net import Mclr_Logistic, DNN

ModelList = {
    'mclr': Mclr_Logistic
    , 'dnn': DNN
}

exp_name = 'hw_2022ml'
data_path = './data/data.npy'
label_path = './data/label.npy'

if __name__ == "__main__":
    datas = np.load(data_path)
    labels = np.load(label_path)
    datas = [(Tensor(x).type(torch.float32), y) for x, y in zip(datas[:], labels[:])]
    iterdata = iter(DataLoader(datas, batch_size=num_train, shuffle=True))
    d_train, l_trian = next(iterdata)
    d_test, l_test = next(iterdata)

    train_data = [(x, y) for x, y in zip(d_train, l_trian)]
    test_data = [(x, y) for x, y in zip(d_test, l_test)]
    trainloader = DataLoader(train_data, batchsize, shuffle=True)
    testloader = DataLoader(test_data, batchsize, shuffle=True)
    itertrain = iter(trainloader)
    # itertest = iter(testloader)

    # trainloader_full = DataLoader(train_data, len(train_data), shuffle=True)
    testloader_full = DataLoader(test_data, len(test_data), shuffle=True)

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    for m in config.model:
        for t in range(times):
            model = ModelList[m]().to(device)
            loss = NLLLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            run = aim.Run(experiment=exp_name + '_' + m)
            run['hparam'] = {
                'num_train': config.num_train,
                'num_test': config.num_test,
                'learning_rate': config.learning_rate,
                'batchsize': config.batchsize,
                'num_epochs': config.num_epochs,
                'model': m
            }

            for i in trange(num_epochs):
                while (True):
                    try:
                        (x, y) = next(itertrain)
                    except StopIteration:
                        itertrain = iter(trainloader)
                        break
                    X, Y = x.to(device), y.to(device)
                    model.train()
                    optimizer.zero_grad()
                    output = model(X)
                    l = loss(output, Y)
                    l.backward()
                    optimizer.step()

                model.eval()

                train_acc = 0
                train_loss = 0
                for x, y in trainloader:
                    X, Y = x.to(device), y.to(device)
                    output = model(X)
                    train_acc += (torch.sum(torch.argmax(output, dim=1) == Y)).item()
                    train_loss += loss(output, Y)
                run.track(train_acc / num_train, 'Training Accuracy(%)', epoch=i)
                run.track(train_loss / num_train, 'Training Loss', epoch=i)

                test_acc = 0
                test_loss = 0
                for x, y in testloader_full:
                    X, Y = x.to(device), y.to(device)
                    output = model(X)
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == Y)).item()
                    test_loss += loss(output, Y)
                run.track(test_acc / num_test, 'Testing Accuracy(%)', epoch=i)
                run.track(test_loss / num_test, 'Testing Loss', epoch=i)
