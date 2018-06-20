import torch

a = torch.ones(2,3, 4,4)
b = torch.ones(2,3, 4,4) * 3
fn_loss = torch.nn.MSELoss(size_average=True, reduce=True)
out=fn_loss(a,b)
print(out)
