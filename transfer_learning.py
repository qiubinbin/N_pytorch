import torch
import torchvision as tv
import matplotlib.pyplot as plt
import copy


def show_image(input, title=None):
    # print(input.shape)
    input = input.numpy().transpose(1, 2, 0)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    input = input * std + mean
    input = input.clip(0, 1)  # 截断函数(0,1)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.pause(5)


# plt.ion()  # 交互模式
transforms = {'train': tv.transforms.Compose([tv.transforms.RandomResizedCrop(224),
                                              tv.transforms.RandomHorizontalFlip(0.5),
                                              tv.transforms.ToTensor(),
                                              tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                              ]),
              'val': tv.transforms.Compose([tv.transforms.Resize((256, 256)),
                                            tv.transforms.CenterCrop(224),
                                            tv.transforms.ToTensor(),
                                            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            ])}
data_set = {i: tv.datasets.ImageFolder(root='hymenoptera_data/' + i, transform=transforms[i]) for i in
            ['train', 'val']}
# print(train_set.classes)
data_loader = {i: torch.utils.data.DataLoader(data_set[i],
                                              batch_size=4,
                                              shuffle=True) for i in
               ['train', 'val']}  # DataLoader会把图片数据格式转换成（C*W*H)，而在用plt.imshow显示时必须转换成（W*H*C）
class_image = data_set['train'].classes
"""测试图片显示"""


# for m, n in iter(data_loader['train']):
#     inputs = tv.utils.make_grid(m)
#     classes = [class_image[i] for i in n]
#     show_image(inputs, classes)


def train_model(model, optimizer, lr_optimizer, criterion, epochs):
    """训练模型"""
    model_state = copy.deepcopy(model.state_dict())  # 最佳模型的状态
    accuracy = 0  # 准确率
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            """每个训练测试阶段重新统计代价和精确度"""
            loss_sum = 0
            corr_sum = 0
            if phase == 'train':
                model.train()
                lr_optimizer.step()
            else:
                model.eval()
            for inputs, labels in data_loader[phase]:
                inputs, labels, model = inputs.cuda(), labels.cuda(), model.cuda()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, pred = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                loss_sum += loss * inputs.size()[0]
                corr_sum += (pred == labels).sum()
            if (phase == 'val') and (corr_sum.double() / len(data_set['val']) > accuracy):
                model_state = copy.deepcopy(model.state_dict())
                accuracy = corr_sum.double() / len(data_set['val'])
                print('New Best Model! Epoch:{} Accuracy:{:0.2f}%'.format(epoch + 1, accuracy * 100))
            print(
                'Loss_Average:{:0.2f} Epoch:{}/{} Phase:{}'.format(float(loss_sum / len(data_set[phase])), epoch + 1,
                                                                   epochs, phase))
    model.load_state_dict(model_state)
    return model


def visualizing(model, image_nums=6):
    """"显示图片"""
    image_sum = 0
    training_save = model.training  # 评估之前保存模型的training状态
    model.eval()  # 进入测试模式
    plt.figure()
    for images, labels in data_loader['val']:
        with torch.no_grad():
            images = images.cuda()
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            labels_pred = [class_image[i] for i in pred]
        for num in range(images.size()[0]):
            image_sum += 1
            ax = plt.subplot(image_nums // 2, 2, image_sum)
            ax.set_title(labels_pred[num])
            ax.axis('off')
            show_image(images.cpu().data[num])
            if image_sum == image_nums:
                model.train(mode=training_save)  # 还原模型状态
                return


"""微调卷积网络，重置最后的全连接层"""
# model = tv.models.resnet18(pretrained=True)  # 载入预训练模型
# in_features = model.fc.in_features
# model.fc = torch.nn.Linear(in_features, 2)  # 重置最后的全连接层
# criterion = torch.nn.CrossEntropyLoss()  # 交叉熵代价函数
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # 随机梯度下降
# lr_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)  # 伽马衰减学习效率优化
# new_model = train_model(model, optimizer, lr_optimizer, criterion, 20)  # 获得最佳模型
# visualizing(new_model)
# plt.ioff()
# plt.show()
"""固定特征提取器"""
model = tv.models.resnet18(pretrained=True)  # 载入预训练模型
for param in model.parameters():  # 冻结卷积层
    param.requires_grad_(False)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 2)  # 重置最后的全连接层
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵代价函数
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.1, momentum=0.9)  # 随机梯度下降,注意和上面的区别
lr_optimizer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)  # 伽马衰减学习效率优化
new_model = train_model(model, optimizer, lr_optimizer, criterion, 20)  # 获得最佳模型
visualizing(new_model)
plt.ioff()
plt.show()
