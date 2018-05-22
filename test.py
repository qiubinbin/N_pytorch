import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.module(x)
        return x


net = Net()
print(net(torch.ones(1,3,32,32)))
print(net._modules, net._parameters)
