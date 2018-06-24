import torchvision as tv
import torch
from PIL import Image
import matplotlib.pyplot as plt

transfer = tv.transforms.Compose([tv.transforms.Resize((255, 255)),
                                  tv.transforms.ToTensor()])
image = Image.open('imgs/cat/cat1.jpg')
out = transfer(image)
plt.figure()  # 直接显示Tensor
plt.imshow(torch.transpose(torch.transpose(out, 0, 1), 1, 2))
plt.figure()  # 先转换成numpy数组再显示
plt.imshow(out.numpy().transpose(1, 2, 0))
plt.show()
