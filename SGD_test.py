import torch
from torch import nn
from torch import optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        # x = torch.nn.functional.relu(self.conv1(x))或者
        x = nn.ReLU()(self.conv1(x))
        # x = torch.nn.functional.relu(self.conv2(x))或者
        x = nn.ReLU()(self.conv2(x))
        return x


net = Net()
# print(list(net.named_parameters()))
optimizer = optim.SGD(params=[{'params': net.conv1.parameters(), 'lr': 1}, {'params': net.conv2.parameters(), 'lr': 2}],
                      lr=0.1)  # 最后一个lr无效？？
print(optimizer)
optimizer.zero_grad()
input = torch.randn(1, 3, 32, 32)
output = net(input).sum()
output.backward()
optimizer.step()
# print(list(net.named_parameters()))
