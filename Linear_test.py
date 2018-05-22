import torch
from torch import nn


class Test(nn.Module):
    def __init__(self, input_features, hidden_features, out_festures):
        super().__init__()
        self.modules = nn.Sequential(
            nn.Linear(input_features, hidden_features), nn.Linear(hidden_features, out_festures))
        # self.layer1 = nn.Linear(input_features, hidden_features)
        # self.layer2 = nn.Linear(hidden_features, out_festures)

    def forward(self, x):
        return self.modules(x)
        # x = self.layer1(x)
        # x = torch.sigmoid(x)
        # x = self.layer2(x)
        # return x


net = Test(3, 4, 1)
data = torch.ones(3).unsqueeze(0)
# print(net(data))
for name, param in net.named_parameters():
    print(name, param.size())
