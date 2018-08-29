from skimage import io, transform
import torch
from torchvision import transforms, utils
import torch.nn.functional as F
import sys
sys.path.append('../')
from src.ImageDataset import *


### whole train dataset
mytransform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4814507, 0.44941443, 0.3985094), (0.2703836, 0.2638982, 0.27239165))
])

## [0.48192835 0.45002526 0.39907682] [0.27025667 0.26409653 0.27254263]
mytransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4814507, 0.45002526, 0.39907682), (0.27025667, 0.26409653, 0.27254263))
])

dataset = ClassificationDataset("../data/DatasetA_train/train", "../data/DatasetA_train/label_list.txt", "../data/DatasetA_train/train.txt", transform=mytransform)

dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=4)

