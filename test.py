import torch
import timeit

def plus(tensor):
    result = []
    for i in tensor:
        result.append(i + 1)
    return torch.Tensor(result)


a = torch.ones(3)
print(timeit.timeit('a+1', globals={'a': a}, number=10000))
print(timeit.timeit('plus(a)', globals={'a': a,'plus':plus}, number=10000))
