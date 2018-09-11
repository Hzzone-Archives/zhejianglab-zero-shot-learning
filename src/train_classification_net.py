import torch.nn as nn
import torch.optim as optim
from torchvision import models
import sys
sys.path.insert(0, ".")

from ImageDataset import *
from train_model import *



if __name__ == "__main__":


    '''
    Training Config
    '''
    BATCH_SIZE = 30
    LR = 0.001
    EPOCH = 100
    Pretrained = False
    NUM_CLASSES = 190

    root = "../DatasetA/"

    dataloaders = getImageLoaders(root, BATCH_SIZE)

    # model_ft = ClassificationModelResenet18(pretrained=Pretrained, NUM_CLASSES=NUM_CLASSES)
    model_ft = models.resnet152(num_classes=NUM_CLASSES)
    # model_ft = models.inception_v3(num_classes=NUM_CLASSES)
    # model_ft = models.resnet18()
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()

    # 如你所见, 所有参数都将被优化
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LR, momentum=0.9)

    model_ft = train_model(model_ft, criterion, optimizer_ft, dataloaders,
                           num_epochs=EPOCH)


