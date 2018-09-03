from src.model import *
from src.utils import *
from src.train_model import *
from src.ImageDataset import *
import torch
from torch import optim

if __name__ == "__main__":
    '''
    Training Config
    '''
    BATCH_SIZE = 100
    LR = 0.001
    EPOCH = 100
    Pretrained = False

    root = "../data/DatasetA_train/"

    dataloaders = getDataLoaders(root, BATCH_SIZE)
    model_ft = torch.load("resnet18.pkl", map_location='cpu')
    featuresModel = FeaturesNet(model_ft)
    DEM_model = DEMNet(featuresModel, 512, 300)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()

    # 如你所见, 所有参数都将被优化
    optimizer_ft = optim.SGD(DEM_model.parameters(), lr=LR, momentum=0.9)

    model_ft = train_model(DEM_model, criterion, optimizer_ft, dataloaders,
                           num_epochs=EPOCH, task="wordembeddings")

