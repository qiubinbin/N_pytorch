import torch
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision as tv
import os


class ImagesData(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()
        imgs = os.listdir(path)
        self.imgs = [os.path.join(path, img) for img in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'cat' in img_path.split('/')[-1] else 0
        img_temp = Image.open(img_path)
        img_tensor = ToTensor()(img_temp)
        return img_tensor, label

    def __len__(self):
        return len(self.imgs)


test=Image.open('imgs/cat1.jpg')
test1=tv.transforms.Resize((224,224))(test)
test2=tv.transforms.Pad()
test2.show()
