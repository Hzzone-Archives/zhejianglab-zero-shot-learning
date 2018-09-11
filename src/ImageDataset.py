from __future__ import print_function, division
import os
from skimage import io, color
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torch
from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class ImageDataset(Dataset):
    def __init__(self, data_root, data_file, vec_file, transform=None):
        self.data_root = data_root
        self.data = np.loadtxt(data_file, delimiter=',', dtype=str)
        self.vec = np.loadtxt(vec_file, delimiter=',', dtype=float)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.data[idx, 0])
        label = int(self.data[idx, 1])
        image = io.imread(img_path)

        ## exists some gray images
        if len(image.shape) != 3:
            image = color.gray2rgb(image)

        # print(image.shape)
        # sample = {"image": image, "label": label}
        if self.transform:
            image = self.transform(image)

        return image, label, self.vec[label]

    def __len__(self):
        return self.data.shape[0]

def getTransforms():

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms

def getImageLoaders(root, BATCH_SIZE=120):

    data_transforms = getTransforms()

    image_datasets = {x: ImageDataset(os.path.join(root, x), os.path.join(root, x+'.csv'), os.path.join(root, "attributes_per_class.csv"), transform=data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                  shuffle=True if x == "train" else False, num_workers=4)
                   for x in ['train', 'val']}
    return dataloaders

