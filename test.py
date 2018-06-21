import torchvision
import torch
from PIL import Image

dataset = torchvision.datasets.ImageFolder('hymenoptera_data/train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
print(len(dataset),len(dataloader))
image=Image.open('hymenoptera_data/train/ants/0013035.jpg')
print(image)