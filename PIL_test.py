from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import torch
from torch import nn

to_tensor = ToTensor()
to_pil = ToPILImage()
lena = Image.open('imgs/xiongmao.jpg')
lena.show()
input = to_tensor(lena).unsqueeze(0)
kernel = torch.ones(3, 3) / -9
kernel[1, 1] = 1
conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)
out = conv(input)
to_pil(out.data.squeeze(0)).show()
pool = nn.AvgPool2d(2, 2)
out1 = pool(input)
to_pil(out1.data.squeeze(0)).show()
