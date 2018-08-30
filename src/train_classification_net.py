from skimage import io, transform
import torch
from torchvision import transforms, utils
import torch.nn.functional as F
import sys
sys.path.append('../')
from src.ImageDataset import *
import time
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import copy
import torch.nn as nn
import torch.optim as optim



### whole train dataset
mytransform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4814507, 0.44941443, 0.3985094), (0.2703836, 0.2638982, 0.27239165))
])

## [0.48192835 0.45002526 0.39907682] [0.27025667 0.26409653 0.27254263]

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Scale(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4814507, 0.45002526, 0.39907682), (0.27025667, 0.26409653, 0.27254263))
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4814507, 0.45002526, 0.39907682), (0.27025667, 0.26409653, 0.27254263))
    ]),
}

# dataset = ClassificationDataset("../data/DatasetA_train/train", "../data/DatasetA_train/label_list.txt", "../data/DatasetA_train/train.txt", transform=mytransform)
#
# dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=4)

'''
Training Config
'''
BATCH_SIZE = 4
LR = 0.001
EPOCH = 300
Pretrained = False
NUM_CLASSES = 190




image_datasets = {x: ClassificationDataset("../data/DatasetA_train/train", \
                                           "../data/DatasetA_train/label_list.txt", "../data/DatasetA_train/{}.txt".format(x),
                                           transform=data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle= True if x == "train" else False, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每一个迭代都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # 设置 model 为训练 (training) 模式
            else:
                model.train(False)  # 设置 model 为评估 (evaluate) 模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for bth_index, data in enumerate(dataloaders[phase]):
                # 获取输入
                inputs, labels = data
                # print(inputs.size(), labels)

                # 用 Variable 包装输入数据
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # 设置梯度参数为 0
                optimizer.zero_grad()

                # 正向传递
                outputs = model(inputs)
                # print(outputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # print(bth_index, loss)

                # 如果是训练阶段, 向后传递和优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 统计
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 深拷贝 model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型的权重
    model.load_state_dict(best_model_wts)
    return model

class ClassificationModelResenet18(nn.Module):
    def __init__(self, pretrained=False, NUM_CLASSES=190):
        super(ClassificationModelResenet18, self).__init__()
        resnet_model = models.resnet18(pretrained=Pretrained)
        for name, module in list(resnet_model._modules.items())[:-2]:
            self.add_module(name, module)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        for name, module in list(self._modules.items())[:-1]:
            x = module(x)
        x = x.view(x.size(0), -1) # 展平多维的卷积图进全连接
        x = self.fc(x)
        return x

# model_ft = models.resnet18(pretrained=Pretrained)
# model_ft = models.alexnet(pretrained=Pretrained)
# model_ft = models.resnet34(pretrained=Pretrained)
model_ft = ClassificationModelResenet18(pretrained=Pretrained, NUM_CLASSES=NUM_CLASSES)


if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# 如你所见, 所有参数都将被优化
optimizer_ft = optim.SGD(model_ft.parameters(), lr=LR, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=EPOCH//4, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=EPOCH)

torch.save(model_ft, "resnet18.pkl")


