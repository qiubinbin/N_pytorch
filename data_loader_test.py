import pandas as pd
import os
from torch.utils.data import Dataset
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torchvision import transforms, utils


def show_landmarks(image, landmarks):
    """显示图片和标记"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='.', c='g')


class FaceMark(Dataset):
    """数据集类"""

    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[item, 0])  # os.path.join对于两个字符串会自动添加'\'
        image = io.imread(img_name)  # H&W&C
        landmarks = self.landmarks_frame.iloc[item, 1:].as_matrix().astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample


test = FaceMark('faces/face_landmarks.csv', 'faces')
print(len(test))
fig = plt.figure()
for i in range(len(test)):
    sample = test[i]
    print(i, sample['image'].shape, sample['landmarks'].shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()  # 紧凑显示图片，居中显示
    ax.set_title('Sample #%s' % i)
    # ax.axis('off')
    show_landmarks(**sample)  # 字典拆分,字典的键与函数形参一样
    if i == 3:
        plt.show()
        break


class Rescale():
    def __init__(self, out_size):
        assert isinstance(out_size, (int, tuple))
        self.out_size = out_size

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.out_size, int):
            if h > w:
                new_h, new_w = self.out_size * h / w, self.out_size
            else:
                new_h, new_w = self.out_size, self.out_size * w / h
        else:
            new_h, new_w = self.out_size
        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w))  # 注意模式默认为constant
        landmark = landmark * [new_w / w, new_h / h]  # 特别注意：横纵交换,坐标点的横纵坐标是反的
        return {'image': image, 'landmarks': landmark}


class RandomCrop():
    """随机裁剪制定大小的图片"""

    def __init__(self, out_size):
        assert isinstance(out_size, (int, tuple))
        if isinstance(out_size, int):
            self.out_size = (out_size, out_size)
        else:
            assert len(out_size) == 2
            self.out_size = out_size

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.out_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top:top + new_h, left:left + new_w]
        landmark = landmark - [left, top]  # 坐标点迁移，没有变形
        return {'image': image, 'landmarks': landmark}


class ToTensor():
    """把数组转换成张量"""

    def __init__(self):
        pass

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmarks']
        # 转置：由numpy数组的H*W*C → Tensor的C*H*W
        image = image.transpose(2, 0, 1)
        return {'image': image, 'landmarks': landmark}


scale = Rescale(256)  # 最短边限定到256
crop = RandomCrop(128)  # 裁剪出128*128
composed = transforms.Compose([Rescale(256), RandomCrop(224)])
fig = plt.figure()
sample = test[65]
for i, trfrm in enumerate([scale, crop, composed]):
    transformed_sample = trfrm(sample)
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(trfrm).__name__)
    show_landmarks(**transformed_sample)
plt.show()
