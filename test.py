import torch


# temp = torch.nn.Linear(3,4)
# print(list(temp.named_parameters())[0])
# torch.nn.init.xavier_normal_(temp.weight)
# print(list(temp.named_parameters())[0])
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temp1 = torch.nn.Parameter(torch.randn(3, 4))
        self.temp2 = torch.nn.Linear(3, 4)

    def forward(self, *input):
        pass


net = Net()
print(net._parameters, '\n-------\n', net._modules)
