import torch
import matplotlib.pyplot as plt

lr = 0.001


def produce(batch_size=8):
    """y=2*x+3"""
    x = torch.rand(batch_size, 1) * 20
    y = x * 2 + (1 + torch.randn(batch_size, 1)) * 3
    return x, y


w = torch.rand(1, 1)
b = torch.zeros(1, 1)
for epoch in range(20000):
    x, y = produce()
    predicted_y = x.mm(w) + b.expand_as(y)
    loss = 0.5 * (predicted_y - y) ** 2
    delta_w = x.t().mm(predicted_y - y)
    delta_b = (predicted_y - y).sum()
    w.sub_(lr * delta_w)
    b.sub_(lr * delta_b)
    if epoch % 2000 == 0:
        x1, y1 = produce(20)
        y_p = x1.mm(w) + b
        plt.scatter(x1.numpy(), y1.numpy(), c='g')
        plt.plot(x1.numpy(), y_p.numpy(), c='r')
        plt.show()
        plt.pause(0.5)
        print(w[0, 0], b[0, 0])
