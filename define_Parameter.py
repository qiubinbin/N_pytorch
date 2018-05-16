import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_features, output_features))
        self.biases = nn.Parameter(torch.randn(output_features))

    def forward(self, x):
        x = x.mm(self.weights)
        x = x + self.biases.expand_as(x)
        return x


data = torch.ones(3).unsqueeze(0)
net = Linear(3, 4)
result = net(data)
print(result)
for name, param in net.named_parameters():
    print(name, param)
