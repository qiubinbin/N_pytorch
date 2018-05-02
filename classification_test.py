import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)  # 尺寸为100*2，对应100个坐标点
y0 = torch.zeros(100)  # 标签
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)  # torch标签必须是LongTensor类型
x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# method1
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super().__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x
#
#
# net1 = Net(2, 10, 2)
# method2,快速搭建神经网络
net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
plt.ion()
plt.show()
for i in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), dim=1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
