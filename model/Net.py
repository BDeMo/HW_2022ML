import torch
from torch import nn
import torch.nn.functional as F

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=607500, output_dim=2):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class DNN(nn.Module):
    def __init__(self, input_dim=607500, mid_dim=100, output_dim=2):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x