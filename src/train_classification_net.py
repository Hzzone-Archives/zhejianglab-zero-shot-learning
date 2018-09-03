import torch.nn as nn
import torch.optim as optim
import sys
sys.path.insert(0, ".")

from src.model import ClassificationModelResenet18
from src.ImageDataset import *
from src.train_model import *



if __name__ == "__main__":


    '''
    Training Config
    '''
    BATCH_SIZE = 100
    LR = 0.001
    EPOCH = 100
    Pretrained = False
    NUM_CLASSES = 190

    root = "../data/DatasetA_train/"

    dataloaders = getDataLoaders(root, BATCH_SIZE)

    model_ft = ClassificationModelResenet18(pretrained=Pretrained, NUM_CLASSES=NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()

    # 如你所见, 所有参数都将被优化
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LR, momentum=0.9)

    model_ft = train_model(model_ft, criterion, optimizer_ft, dataloaders,
                           num_epochs=EPOCH)


