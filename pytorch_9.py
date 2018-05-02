import torch as t

x = t.ones(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)
gradients = t.tensor([0.1, 1.0, 0.0001], dtype=t.float)
y.backward(gradients)
print(x.grad)