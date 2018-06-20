import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np
from torch import optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def show(img):  # 显示图片
    img = (img + 1) / 2  # 去归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = transforms.Compose([transforms.ToTensor(),  # 图片转Tensor
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化Tensor
                                ])
"""训练集"""
trainset = tv.datasets.CIFAR10(root='E:/N_pytorch',  # 下载目录
                               train=True,  # 是否从训练集获取数据
                               download=True,  # 是否下载
                               transform=transform,  # 处理
                               )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
"""测试集"""
testset = tv.datasets.CIFAR10(root='E:/N_pytorch',
                              train=False,
                              download=True,
                              transform=transform, )
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# """测试训练集"""
# data, label = trainset[100]
# print(classes[label])
# show(data)
# """测试训练数据batch"""
# datas, labels = iter(trainloader).next()
# print(''.join('%-11s' % classes[labels[i]] for i in range(4)))
# show(tv.utils.make_grid(datas))
net = Net_test()
net.to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵代价函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        """前向传播+反向传播"""
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  # 更新参数
        """打印信息"""
        running_loss += loss.data
        if i % 2000 == 1999:
            print('[%d,%5d] loss: %0.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
# """单个batch测试"""
# """实际情况"""
# dataiter = iter(testloader)
# images, labels = next(dataiter)
# print('实际的label: ', ''.join('%-8s' % classes[labels[i]] for i in range(len(labels))))
# show(tv.utils.make_grid(images))
# """预测情况"""
# outputs = net(images)
# predicted = torch.argmax(outputs, dim=1)
# print('预测的label: ', ''.join('%-8s' % classes[temp] for temp in predicted))
"""测试钩子"""


def forward_hook(self, input, output):
    print('input1: ', input[0].size(), type(input[0]))
    print('output1: ', output.size(), type(output))


def backward_hook(self, input, output):
    print('input2: ', input[0].size(), type(input))
    print('output2: ', output.size(), type(output))


"""整个测试数据集的预测水平"""
correct = 0
total = 0
net.conv1.register_forward_hook(forward_hook)
net.conv2.register_backward_hook(backward_hook)
for data in testloader:
    images, labels = data
    images, labels = images.to(device), labels.to(device)  # 转移到GPU运算，但是注意不要写成images.to(device)
    outputs = net(images)
    predicted = torch.argmax(outputs, dim=1)
    total += len(labels)
    correct += (predicted == labels).sum()
print('10000张测试及数据：%d %%' % (100 * correct / total))
